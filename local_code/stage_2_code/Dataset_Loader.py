'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading Stage 2 data...")

        train_path = "data/stage_2_data/train.csv"
        test_path = "data/stage_2_data/test.csv"

        train_data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
        test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)

        # First column contains digit labels
        y_train = train_data[:, 0].astype(np.int64)
        y_test = test_data[:, 0].astype(np.int64)

        # Remaining 784 columns contain pixel values
        X_train = train_data[:, 1:] / 255.0
        X_test = test_data[:, 1:] / 255.0

        print("Training data:", X_train.shape)
        print("Testing data:", X_test.shape)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }