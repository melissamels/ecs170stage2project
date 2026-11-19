'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
from local_code.base_class.dataset import dataset
import os


class RNN_Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading RNN data...")

        # Dataset folder paths
        train_path = 'data/stage_4_data/text_classification/train'
        test_path = 'data/stage_4_data/text_classification/test'

        x_train = []
        y_train = []

        # Load training data
        for sentiment in ['pos', 'neg']:

            folder_path = os.path.join(train_path, sentiment)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    f = open(os.path.join(folder_path, filename), 'r', encoding='utf-8')

                    review = f.read()
                    f.close()

                    x_train.append(review)
                    y_train.append(1 if sentiment == 'pos' else 0)

        x_test = []
        y_test = []

        # Load testing data

        for sentiment in ['pos', 'neg']:

            folder_path = os.path.join(test_path, sentiment)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    f = open(os.path.join(folder_path, filename), 'r', encoding='utf-8')

                    review = f.read()
                    f.close()

                    x_test.append(review)
                    y_test.append(1 if sentiment == 'pos' else 0)

        # make it a np array
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print("Training data:", len(x_train))
        print("Testing data:", len(x_test))

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }