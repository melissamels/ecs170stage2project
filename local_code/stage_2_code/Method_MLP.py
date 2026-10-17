'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_MLP(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 20
        self.learning_rate = 0.001
        self.loss_history = []
        self.accuracy_history = []

        # MLP architecture: 784 -> 128 -> 64 -> 10
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

    def train_model(self, X, y):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        loss_function = nn.CrossEntropyLoss()

        self.network.train()

        for epoch in range(self.max_epoch):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                outputs = self.forward(batch_X)
                loss = loss_function(outputs, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch_y).sum().item()
                total += len(batch_y)

            avg_loss = total_loss / total
            accuracy = correct / total

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{self.max_epoch} "
                f"Loss: {avg_loss:.4f} "
                f"Accuracy: {accuracy:.4f}"
            )

    def test(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)

        self.network.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            predictions = outputs.argmax(dim=1)

        return predictions.numpy()

    def run(self):
        print("Starting training...")

        self.train_model(
            self.data["X_train"],
            self.data["y_train"]
        )

        print("Starting testing...")

        predictions = self.test(self.data["X_test"])

        return {
            "pred_y": predictions,
            "true_y": self.data["y_test"]
        }
