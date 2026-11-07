import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from local_code.stage_3_code.MNIST_Dataset_Loader import MNIST_Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST

PROJECT_ROOT = Path(__file__).resolve().parents[2] #setting the wd to be ecs170stage2proj
os.chdir(PROJECT_ROOT)

def main():
    np.random.seed(2)
    torch.manual_seed(2)

    print("************ Start Stage 3 - MNIST ************")

    # Load the already separated training and testing data
    data_obj = MNIST_Dataset_Loader("MNIST", "")
    data = data_obj.load()

    # Initialize and train the CNN
    method_obj = Method_CNN_MNIST("MNIST CNN", "")
    method_obj.data = data

    results = method_obj.run()

    #training stats
    y_true = np.asarray(results["train_true_y"])
    y_pred = np.asarray(results["train_pred_y"])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n************ Training Results ************")
    print(f"Accuracy:  {accuracy:.8f}")
    print(f"Precision: {precision:.8f}")
    print(f"Recall:    {recall:.8f}")
    print(f"F1 Score:  {f1:.8f}")
    print("\nTraining Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Evaluate predictions
    y_true = np.asarray(results["true_y"])
    y_pred = np.asarray(results["pred_y"])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n************ Testing Results ************")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


    # Create the results folder
    result_dir = Path("result/stage_3_result/mnist")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    np.savetxt(
        result_dir / "mnist_predictions.csv",
        np.column_stack((y_true, y_pred)),
        delimiter=",",
        fmt="%d",
        header="true_label,predicted_label",
        comments=""
    )

    # Plot learning convergence curves
    epochs = range(1, len(method_obj.loss_history) + 1)

    #loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, method_obj.loss_history, label = "Training Loss")
    plt.plot(epochs, method_obj.test_loss_history, label = "Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST CNN Training vs. Testing Loss")
    plt.grid(True)
    plt.savefig(result_dir / "MNIST_loss.png")
    plt.close()

    #accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, method_obj.accuracy_history, label = "Training Accuracy")
    plt.plot(epochs, method_obj.test_accuracy_history, label = "Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Training vs. Testing Accuracy")
    plt.grid(True)
    plt.savefig(result_dir / "MNIST_accuracy.png")
    plt.close()

    print("\nResults saved to:", result_dir)
    print("************ Finish ************")


if __name__ == "__main__":
    main()