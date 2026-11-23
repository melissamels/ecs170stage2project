'''
Concrete IO class for a specific dataset
'''
import random

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
from local_code.base_class.dataset import dataset
import os
import csv

class RNN_Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading RNN GEN data...")

        # Dataset folder paths
        data_path = 'data/stage_4_data/text_generation/data'

        joke = []

        # Load training data
        with open(data_path, 'r', encoding='utf-8', newline = '') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["Joke"].strip()
                if text:
                    joke.append(text)

            random.seed(1)
            random.shuffle(joke)
            index = int(len(joke) * 0.8) #80/20 training split

        x_train = joke[:index] #first chunk
        x_test = joke[index:] #second chunk
        #
        # # make it a np array
        # y_train = np.array(y_train)
        # y_test = np.array(y_test)

        print("Training data:", len(x_train))
        print("Testing data:", len(x_test))

        return {
            "x_train": x_train,
            # "y_train": y_train,
            "x_test": x_test
            #"y_test": y_test
        }