'''
Concrete MethodModule class for a specific learning MethodModule
'''
import string
import os
from pathlib import Path
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2] #setting the wd to be ecs170stage2proj
os.chdir(PROJECT_ROOT)

class Method_RNN_gen(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 50
        self.learning_rate = 0.0005
        self.batch_size = 50

        self.loss_history = []
        self.accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

        self.vocab_size = None # Max number of unique words (tokens)
        self.embedding_dim = 300 # Number of values used to represent each word
        self.hidden_size = 200 # Number of features maintained in the RNN hidden state

        # the layers
        self.dropout = nn.Dropout(0.3) # randomly turns off values to reduce overfitting
        self.embedding = None
        self.rnn = None
        self.max_seq_length = 150
        self.fc = None
        self.glove_text = "data/stage_4_data/text_generation/glove.txt"
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def load_glove_embeddings(self):
        # random embeddings for each word
        embedding_matrix = torch.empty(self.vocab_size, self.embedding_dim)
        nn.init.normal_(embedding_matrix, mean=0, std=0.1)

        # padding token = zero
        embedding_matrix[0] = torch.zeros(self.embedding_dim)
        found = 0

        # only load vectors for words in vocab
        vocabulary = {}
        for word, idx in self.tokenizer.word_index.items():
            clean_word = word.strip(string.punctuation)
            if clean_word:
                vocabulary.setdefault(clean_word, []).append(idx)

        with open(self.glove_text, "r", encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word not in vocabulary:
                    continue

                if len(values) != self.embedding_dim + 1:
                    continue

                try:
                    vector_values = [float(x) for x in values[1:]]
                except ValueError:
                    continue

                vector = torch.tensor(vector_values, dtype=torch.float)
                for idx in vocabulary[word]:
                    embedding_matrix[idx] = vector
                    found += 1

        print(
            f"Loaded {found}/{self.vocab_size - 1} "
            f"words from glove"
        )
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )

    def forward(self, x):
        emb = self.embedding(x) # word IDs -> vectors

        output, (hidden, cell) = self.rnn(emb) # RNN reads the whole review
        #last = hidden[-1] # memory after the LAST word
        return self.fc(self.dropout(output)) # -> 2 scores #self.dropout(last)

    def clean_text(self, text):
        import re, html
        text = html.unescape(text)
        #text = re.sub(r'\.{2,}', '.', text)
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'([.,!?;:()"])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        #text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()

    def build_tokenizer(self, x_train):
        jokes = ["<start> " + self.clean_text(joke) + " <end>" for joke in x_train]
        self.tokenizer = Tokenizer(filters="", lower=True, oov_token="<unk>")
        self.tokenizer.fit_on_texts(jokes)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Vocabulary size:", self.vocab_size)
        self.load_glove_embeddings()
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def make_loader(self, x, shuffle=True):
        jokes = ["<start> " + self.clean_text(joke) + " <end>" for joke in x]
        sequences = self.tokenizer.texts_to_sequences(jokes)

        inputs = []
        targets = []

        for sequence in sequences:
            sequence = sequence[:self.max_seq_length]

            if len(sequence) > 1:
                inputs.append(
                    torch.tensor(sequence[:-1], dtype=torch.long)
                )
                targets.append(
                    torch.tensor(sequence[1:], dtype=torch.long)
                )

        # pad sequences to same length
        inputs = pad_sequence(inputs,batch_first=True, padding_value=0)
        targets = pad_sequence(targets, batch_first=True, padding_value=0)
        dataset = TensorDataset(inputs, targets)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def generate_text(self, prompt, max_words=60, temperature=0.8):
        self.eval()
        prompt = self.clean_text(prompt)

        start_token = self.tokenizer.word_index["<start>"]
        end_token = self.tokenizer.word_index["<end>"]

        prompt_tokens = self.tokenizer.texts_to_sequences([prompt])[0]
        generated = [start_token] + prompt_tokens

        with torch.no_grad():
            for _ in range(max_words):
                context = generated[-self.max_seq_length:]
                x = torch.tensor(
                    [generated[-self.max_seq_length:]],
                    dtype=torch.long
                )
                outputs = self(x)
                logits = outputs[0, -1, :] / temperature

                logits[0] = float("-inf")
                logits[start_token] = float("-inf")

                top_k = 20 #look at the top 20 words
                top_logits, top_indices = torch.topk(logits, top_k)
                probabilities = torch.softmax(top_logits, dim=-1)
                sampled_index = torch.multinomial(probabilities, 1)
                next_token = top_indices[sampled_index].item()

                # next_token = torch.argmax(logits).item() #take the best one

                if next_token == end_token: #stop if end token
                    break
                generated.append(next_token)

        words = [
            self.tokenizer.index_word.get(token, "<unk>")
            for token in generated
            if token not in (start_token, 0)
        ]

        return " ".join(words)

    def train_model(self, data):
        self.build_tokenizer(data["x_train"])
        self.trainloader = self.make_loader(data["x_train"], True)
        self.testloader = self.make_loader(data["x_test"], False)

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )

        best_test_loss = float("inf")
        best_model = None
        for epoch in range(self.max_epoch):
            self.train()

            total_loss = 0
            total_tokens = 0

            for inputs, targets in self.trainloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs.reshape(-1, self.vocab_size), targets.reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                valid_tokens = (targets != 0).sum().item()

                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

            train_loss = total_loss / total_tokens
            test_loss = self.test(self.testloader)

            if test_loss < best_test_loss: #get the best loss
                best_test_loss = test_loss
                best_model = {
                    key: value.clone()
                    for key, value in self.state_dict().items()
                }

            self.loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch} "
                f"Train Loss: {train_loss:.4f} "
                f"Test Loss: {test_loss:.4f}"
            )

        self.load_state_dict(best_model)
        print("Best test loss:", best_test_loss)

    def test(self, loader):
        self.eval()

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self(inputs)
                loss = self.criterion(
                    outputs.reshape(-1, self.vocab_size),
                    targets.reshape(-1)
                )
                valid_tokens = (targets != 0).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

        return total_loss / total_tokens

    def run(self):

        print("Starting training...")

        self.train_model(self.data)

        print("\nStarting joke generation...")

        return {
            "loss_history": self.loss_history,
            "test_loss_history": self.test_loss_history
        }