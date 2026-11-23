'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import csv
import random
import numpy as np
from local_code.base_class.dataset import dataset
import os


class RNN_Gen_Dataset_Loader(dataset):

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading RNN data...")

        data_path = 'data/stage_4_data/text_generation/data'  # path to the data

        jokes = []

        f = open(data_path, 'r', encoding='utf-8-sig', newline='')
        for row in csv.DictReader(f):  # header: "ID","Joke"
            jokes.append(row["Joke"].strip())
        f.close()

        #shuffle and split 80/20
        random.Random(1).shuffle(jokes)
        split = int(len(jokes) * 0.8)
        x_train = jokes[:split]
        x_test = jokes[split:]

        print("Training data:", len(x_train))
        print("Testing data:", len(x_test))

        return {
            "x_train": x_train,
            "x_test": x_test
        }