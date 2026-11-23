'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import torch.nn.functional as F

from tensorflow.keras.preprocessing.text import Tokenizer

class Method_RNN_Gen(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 30
        self.learning_rate = 0.001
        self.batch_size = 64

        self.loss_history = []
        self.accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

        self.vocab_size = 5000 # Max number of unique words (tokens)
        self.embedding_dim = 64 # Number of values used to represent each word
        self.hidden_size = 128 # Number of features maintained in the RNN hidden state
        self.num_layers = 1 # Number of layers
        

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3) # Randomly turn off values to reduce overfitting
        self.fc = nn.Linear(self.hidden_size, self.vocab_size) 

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x):
        emb = self.embedding(x) # word IDs -> vector
        output, _ = self.rnn(emb)
        return self.fc(self.dropout(output))

    def step(self, x, hidden=None):  # used only by generate
        emb = self.embedding(x)
        output, hidden = self.rnn(emb, hidden)
        return self.fc(output), hidden

    def build_tokenizer(self, x_train):
        filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<UNK>", filters=filters)
        self.tokenizer.fit_on_texts([text + " <EOS>" for text in x_train])
        
        self.UNK = self.tokenizer.word_index["<UNK>"]
        self.EOS = self.tokenizer.word_index["<eos>"]

    def make_loader(self, x, shuffle):
        x = self.tokenizer.texts_to_sequences(x)

        x = [review[:500] + [self.EOS] for review in x]

        y = [torch.tensor(review[1:], dtype=torch.long) for review in x]
        x = [torch.tensor(review[:-1], dtype=torch.long) for review in x]
        
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)

        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=shuffle)



    def train_model(self, data):

        # Build tokenizer using training reviews only
        self.build_tokenizer(data["x_train"])

        self.trainloader = self.make_loader(data["x_train"], True)
        self.train_eval_loader = self.make_loader(data["x_train"], False)
        self.testloader = self.make_loader(data["x_test"], False)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Training loop for each epoch
        for epoch in range(self.max_epoch):
            self.train()

            for inputs, labels in self.trainloader:
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs.reshape(-1, self.vocab_size), labels.reshape(-1))

                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), 5.0)

                optimizer.step()

            _, train_loss, train_accuracy = self.test(self.train_eval_loader)
            _, test_loss, test_accuracy = self.test(self.testloader)

            self.loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)
            self.accuracy_history.append(train_accuracy)
            self.test_accuracy_history.append(test_accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch} "
                f"Loss: {train_loss:.4f} "
                f"Accuracy: {train_accuracy:.4f}"
            )

    def test(self, loader):
        self.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        predictions = []

        with (torch.no_grad()): # dont need grads while testing 
            for inputs, labels in loader: 
                outputs = self(inputs)
                predicted = outputs.argmax(dim=-1) # pick the most likely next token
                mask = labels != 0
                predictions.extend(predicted[mask].numpy())
                total += mask.sum().item()
                total_loss += self.criterion(outputs.reshape(-1, self.vocab_size), labels.reshape(-1)).item() * mask.sum().item()
                correct += ((predicted == labels) & mask).sum().item()

        epoch_loss = total_loss / total
        epoch_test_accuracy = correct / total

        return predictions, epoch_loss, epoch_test_accuracy
    

    @torch.no_grad()
    def generate(self, prompt, temperature=1.0):
        self.eval()

        ids = self.tokenizer.texts_to_sequences([prompt])[0]
        x = torch.tensor([ids], dtype=torch.long)

        score, hidden = self.step(x)
        score = score[:, -1, :]  # distribution after the last token
        generated = list(ids)

        for _ in range(50):
            score = score / temperature
            score[:, [0, self.UNK]] = float("-inf")  # never sample index 0 or <UNK>

            probs = F.softmax(score, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() == self.EOS:
                break
            generated.append(next_id.item())

            scores, hidden = self.step(next_id, hidden)
            scores = scores[:, -1, :]

        return " ".join(self.tokenizer.index_word.get(id, "<UNK>") for id in generated)

    def run(self):
        print("Starting training...")

        self.train_model(
            self.data
        )

        train_predictions, _, _ = self.test(self.train_eval_loader)

        print("Starting testing...")

        predictions, _, _ = self.test(self.testloader)

        train_y = torch.cat([y[y != 0] for _, y in self.train_eval_loader]).numpy()
        test_y = torch.cat([y[y != 0] for _, y in self.testloader]).numpy()

        return {
            "train_pred_y": train_predictions,
            "train_true_y": train_y,
            "pred_y": predictions,
            "true_y": test_y
        }