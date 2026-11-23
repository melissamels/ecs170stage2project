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

from tensorflow.keras.preprocessing.text import Tokenizer

class Method_LSTM(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 10
        self.learning_rate = 0.001
        self.batch_size = 256

        self.loss_history = []
        self.accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []

        self.vocab_size = 10000 # Max number of unique words (tokens)
        self.embedding_dim = 130 # Number of values used to represent each word
        self.hidden_size = 256 # Number of features maintained in the RNN hidden state

        # the layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn1 = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.5) # randomly turns off values to reduce overfitting
        self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)  # randomly turns off values to reduce overfitting
        self.fc = nn.Linear(self.hidden_size, 2) # 2 classes: negative / positive

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        emb = self.embedding(x) # word IDs -> vectors

        output, (hidden, cell) = self.rnn1(emb) # RNN reads the whole review
        output = self.dropout1(output)
        output, (hidden, cell) = self.rnn2(output) # RNN reads the whole review
        summed = torch.sum(output, dim=1)
        # last = hidden[-1] # memory after the LAST word
        return self.fc(self.dropout2(summed)) # -> 2 scores

    def build_tokenizer(self, x_train):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<UNK>")
        self.tokenizer.fit_on_texts(x_train)

    def make_loader(self, x, y, shuffle):
        x = self.tokenizer.texts_to_sequences(x)

        x = [review[:150] for review in x]

        x = [torch.tensor(review, dtype=torch.long) for review in x]
        x = pad_sequence(x, batch_first=True, padding_value=0)

        y = torch.tensor(y, dtype=torch.long)

        return DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=shuffle)

    def train_model(self, data):

        # Build tokenizer using training reviews only
        self.build_tokenizer(data["x_train"])

        self.trainloader = self.make_loader(data["x_train"], data["y_train"], True)
        self.train_eval_loader = self.make_loader(data["x_train"], data["y_train"], False)
        self.testloader = self.make_loader(data["x_test"], data["y_test"], False)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Training loop for each epoch
        for epoch in range(self.max_epoch):
            self.train()

            for inputs, labels in self.trainloader:
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

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
                predicted = outputs.argmax(dim=1) # pick the higher score
                predictions.extend(predicted.numpy())
                total += labels.size(0)
                total_loss += self.criterion(outputs, labels).item() * labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = total_loss / total
        epoch_test_accuracy = correct / total

        return predictions, epoch_loss, epoch_test_accuracy

    def run(self):
        print("Starting training...")

        self.train_model(
            self.data
        )

        train_predictions, _, _ = self.test(self.train_eval_loader)

        print("Starting testing...")

        predictions, _, _ = self.test(self.testloader)

        return {
            "train_pred_y": train_predictions,
            "train_true_y": self.data["y_train"],
            "pred_y": predictions,
            "true_y": self.data["y_test"]
        }