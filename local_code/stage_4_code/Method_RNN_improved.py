
'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from local_code.base_class.method import method


class Method_RNN(method, nn.Module):
    """Bidirectional LSTM text classifier with validation-based early stopping."""

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.max_epoch = 10
        self.learning_rate = 0.001
        self.batch_size = 64
        self.vocab_size = 20000
        self.embedding_dim = 100
        self.hidden_size = 128
        self.max_length = 500
        self.patience = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_history = []
        self.accuracy_history = []
        # These hold VALIDATION metrics, not test metrics, to avoid test-set tuning.
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.validation_loss_history = self.test_loss_history
        self.validation_accuracy_history = self.test_accuracy_history

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_size * 2, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = (x != 0).sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.rnn(packed)
        # Last forward and backward hidden states of final BiLSTM layer.
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))

    def build_tokenizer(self, x_train):
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<UNK>")
        self.tokenizer.fit_on_texts(list(x_train))

    def make_loader(self, x, y, shuffle):
        sequences = self.tokenizer.texts_to_sequences(list(x))
        sequences = [s[:self.max_length] or [1] for s in sequences]
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        max_len = max(1, max(map(len, sequences)))
        padded = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        labels = torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)
        return DataLoader(
            TensorDataset(padded, lengths, labels), batch_size=self.batch_size,
            shuffle=shuffle, num_workers=0
        )

    def test(self, loader):
        self.eval()
        total, total_loss, correct = 0, 0.0, 0
        predictions = []
        with torch.no_grad():
            for inputs, lengths, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs, lengths)
                pred = outputs.argmax(dim=1)
                predictions.extend(pred.cpu().tolist())
                batch_size = labels.size(0)
                total += batch_size
                total_loss += self.criterion(outputs, labels).item() * batch_size
                correct += (pred == labels).sum().item()
        return predictions, total_loss / total, correct / total

    def train_model(self, data):
        torch.manual_seed(42)
        np.random.seed(42)
        x = np.asarray(data["x_train"], dtype=object)
        y = np.asarray(data["y_train"], dtype=np.int64)
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.1, random_state=42, stratify=y
        )
        self.build_tokenizer(x[train_idx])
        self.trainloader = self.make_loader(x[train_idx], y[train_idx], True)
        self.train_eval_loader = self.make_loader(x[train_idx], y[train_idx], False)
        self.validloader = self.make_loader(x[val_idx], y[val_idx], False)
        self.testloader = self.make_loader(data["x_test"], data["y_test"], False)

        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(self.max_epoch):
            self.train()
            for inputs, lengths, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self(inputs, lengths), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

            _, train_loss, train_acc = self.test(self.train_eval_loader)
            _, val_loss, val_acc = self.test(self.validloader)
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)
            self.test_loss_history.append(val_loss)
            self.test_accuracy_history.append(val_acc)
            print(f"Epoch {epoch+1}/{self.max_epoch} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_loss - 0.0001:
                best_loss = val_loss
                best_state = deepcopy(self.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print("Early stopping based on validation loss")
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    def run(self):
        self.train_model(self.data)
        train_predictions, _, _ = self.test(self.train_eval_loader)
        predictions, test_loss, test_accuracy = self.test(self.testloader)
        print(f"Final held-out test accuracy: {test_accuracy:.4f}; loss: {test_loss:.4f}")
        return {
            "train_pred_y": train_predictions,
            "train_true_y": self.train_eval_loader.dataset.tensors[2].numpy(),
            "pred_y": predictions,
            "true_y": self.data["y_test"]
        }

