'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from torch import nn
import torch
import torchvision
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Method_CNN_MNIST(method, nn.Module):

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 20
        self.learning_rate = 0.001
        self.loss_history = []
        self.accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.batch_size = 32

        # original dim: 28 x 28
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.trainloader = None
        self.testloader = None

    def forward(self, x):
        return self.network(x)

    def train_model(self, data):

        #convert to pytorch tensors
        x_train = torch.tensor(data["x_train"], dtype=torch.float32)
        y_train = torch.tensor(data["y_train"], dtype=torch.long)
        x_test = torch.tensor(data["x_test"], dtype=torch.float32)
        y_test = torch.tensor(data["y_test"], dtype=torch.long)

        #manually normalize
        x_train = (x_train / 255.0 - 0.5) / 0.5
        x_test = (x_test / 255.0 - 0.5) / 0.5

        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)  # add channel dim + reorder correctly
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

        trainset = TensorDataset(x_train, y_train) #make datasets using x and y tensors
        testset = TensorDataset(x_test, y_test)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)

        self.train_eval_loader = DataLoader(trainset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=2) #unshuffled set for eval

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=2) #dont need to shuffle test

        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        #training loop for each epoch
        for epoch in range(self.max_epoch):
            self.network.train()

            for i, data in enumerate(self.trainloader, 0): #loop for each batch
                inputs, labels = data # gets inputs --> data is a list of [inputs, labels]
                optimizer.zero_grad() #zero param gradient
                #forward + backward + optimize
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
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

    def test(self, testloader):
        self.network.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        predictions = []

        with torch.no_grad(): #dont need grads while testing
            for data in testloader:
                images, labels = data
                # calc outputs
                outputs = self.network(images)
                # pick class w/ highest output
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.numpy())
                total += labels.size(0)
                loss = self.criterion(outputs, labels) #record loss
                test_loss += loss.item() * labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = test_loss / total
        epoch_test_accuracy = correct / total

        return predictions, epoch_loss, epoch_test_accuracy

    def run(self):
        print("Starting training...")

        self.train_model(
            self.data
        )
        #print("Starting training eval...")
        train_predictions, _, _ = self.test(self.train_eval_loader)

        predictions, _, _ = self.test(self.testloader)

        return {
            "train_pred_y": train_predictions,
            "train_true_y": self.data["y_train"],
            "pred_y": predictions,
            "true_y": self.data["y_test"]
        }