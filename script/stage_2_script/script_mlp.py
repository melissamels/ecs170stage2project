import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP


def main():
    np.random.seed(2)
    torch.manual_seed(2)

    print("************ Start Stage 2 ************")

    # Load the already separated training and testing data
    data_obj = Dataset_Loader("handwritten digits", "")
    data = data_obj.load()

    # Initialize and train the MLP
    method_obj = Method_MLP("multi-layer perceptron", "")
    method_obj.data = data

    results = method_obj.run()

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
    result_dir = Path("result/stage_2_result")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    np.savetxt(
        result_dir / "predictions.csv",
        np.column_stack((y_true, y_pred)),
        delimiter=",",
        fmt="%d",
        header="true_label,predicted_label",
        comments=""
    )

    # Plot learning convergence curves
    epochs = range(1, len(method_obj.loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, method_obj.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MLP Training Loss")
    plt.grid(True)
    plt.savefig(result_dir / "training_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, method_obj.accuracy_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("MLP Training Accuracy")
    plt.grid(True)
    plt.savefig(result_dir / "training_accuracy.png")
    plt.close()

    print("\nResults saved to:", result_dir)
    print("************ Finish ************")


if __name__ == "__main__":
    main()