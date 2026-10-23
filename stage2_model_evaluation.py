
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP

np.random.seed(2)
torch.manual_seed(2)

data = Dataset_Loader("handwritten digits", "").load()

model = Method_MLP("multi-layer perceptron", "")

X_train = torch.tensor(np.asarray(data["X_train"]), dtype=torch.float32)
y_train = torch.tensor(np.asarray(data["y_train"]), dtype=torch.long)
X_test = torch.tensor(np.asarray(data["X_test"]), dtype=torch.float32)
y_test = torch.tensor(np.asarray(data["y_test"]), dtype=torch.long)

loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

def evaluate(X, y):
    model.eval()
    total_loss = 0
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), 1024):
            xb = X[start:start + 1024]
            yb = y[start:start + 1024]
            outputs = model(xb)
            total_loss += criterion(outputs, yb).item() * len(yb)
            predictions.extend(outputs.argmax(dim=1).numpy())

    predictions = np.array(predictions)
    return total_loss / len(y), accuracy_score(y.numpy(), predictions), predictions

for epoch in range(20):
    model.train()

    for xb, yb in loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

    train_loss, train_acc, _ = evaluate(X_train, y_train)
    test_loss, test_acc, _ = evaluate(X_test, y_test)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1}/20 | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

out = Path("result/stage_2_result/model_evaluation")
out.mkdir(parents=True, exist_ok=True)

epochs = range(1, 21)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, test_losses, label="Testing Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("MLP Training and Testing Loss")
plt.xticks(range(1, 21))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out / "loss_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, test_accuracies, label="Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MLP Training and Testing Accuracy")
plt.xticks(range(1, 21))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out / "accuracy_comparison.png", dpi=300)
plt.close()

with open(out / "metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Accuracy", "Precision", "Recall", "F1"])

    for name, X, y in [
        ("Training", X_train, y_train),
        ("Testing", X_test, y_test)
    ]:
        _, _, pred = evaluate(X, y)
        true = y.numpy()

        values = [
            accuracy_score(true, pred),
            precision_score(true, pred, average="weighted", zero_division=0),
            recall_score(true, pred, average="weighted", zero_division=0),
            f1_score(true, pred, average="weighted", zero_division=0)
        ]

        writer.writerow([name] + values)
        print(f"{name}: Accuracy={values[0]:.4f}, Precision={values[1]:.4f}, Recall={values[2]:.4f}, F1={values[3]:.4f}")

print("All results saved to:", out)
