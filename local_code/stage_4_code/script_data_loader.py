import pickle
import os
from pathlib import Path
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # setting the wd to be ecs170stage2proj
os.chdir(PROJECT_ROOT)

# loading ORL dataset
if 1:
    from local_code.dataset.RNN_Dataset_Loader import RNN_Dataset_Loader

    loader = RNN_Dataset_Loader()
    data = loader.load()

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # Display one training example
    print("Training review")
    print(x_train[0])
    print("Training label:", y_train[0])

    # Display one testing example
    print("\nTesting review:")
    print(x_test[0])
    print("Testing label:", y_test[0])






