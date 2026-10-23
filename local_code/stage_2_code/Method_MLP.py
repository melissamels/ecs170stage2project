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
        self.test_loss_history = []

        # MLP architecture: 784 -> 128 -> 64 -> 10
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.network(x)

    def train_model(self, X, y, x_test, y_test):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.long)

        #saving the test loss as well
        x_test = torch.tensor(np.asarray(x_test), dtype=torch.float32)
        y_test = torch.tensor(np.asarray(y_test), dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(self.max_epoch):
            self.network.train()

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

            #eval the testing loss
            self.network.eval()
            with torch.no_grad():
                test_outputs = self.forward(x_test)
                test_loss = loss_function(test_outputs, y_test).item()

            self.test_loss_history.append(test_loss)

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
            self.data["y_train"],
            self.data["X_test"],
            self.data["y_test"]
        )
        print("Starting training eval...")
        train_predictions = self.test(self.data["X_train"])

        print("Starting testing...")

        predictions = self.test(self.data["X_test"])

        return {
            "train_pred_y": train_predictions,
            "train_true_y": self.data["y_train"],
            "pred_y": predictions,
            "true_y": self.data["y_test"]
        }
