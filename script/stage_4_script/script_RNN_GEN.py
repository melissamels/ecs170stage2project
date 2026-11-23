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

from local_code.stage_4_code.RNN_Gen_Dataset_Loader import RNN_Dataset_Loader
from local_code.stage_4_code.Method_RNN_gen import Method_RNN_gen

PROJECT_ROOT = Path(__file__).resolve().parents[2] #setting the wd to be ecs170stage2proj
os.chdir(PROJECT_ROOT)

def main():
    np.random.seed(2)
    torch.manual_seed(2)

    print("************ Start Stage 4 - RNN Classification ************")
    print("Add GloVe 300d Wiki: data/stage_4_data/text_generation/glove.txt")
    # Load the already separated training and testing data
    data_obj = RNN_Dataset_Loader("RNN", "")
    data_obj.dataset_source_folder_path = "data/stage_4_data/text_generation/"
    data = data_obj.load()

    # Initialize and train the RNN
    method_obj = Method_RNN_gen("RNN Text Classification", "")
    method_obj.data = data

    res = method_obj.run()

    prompt = [
        "Two men walk",
        "Why did the"]

    generated_jokes = {}

    for p in prompt:
        generated_joke = method_obj.generate_text(
            p,
            max_words=60,
            temperature=0.8
        )

        print("\nStarting words:", p)
        print("Generated joke:", generated_joke)

        generated_jokes[p] = generated_joke

    res["generated_jokes"] = generated_jokes

    # Create the results folder
    result_dir = Path("result/stage_4_result/text_generation")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(
        result_dir / "generated_jokes.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for prompt, joke in res["generated_jokes"].items():

            f.write("Prompt: " + prompt + "\n")
            f.write("Generated: " + joke + "\n\n")

if __name__ == "__main__":
    main()