'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
from local_code.base_class.dataset import dataset
import pickle


class CIFAR_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print("Loading CIFAR data...")
        f = open('data/stage_3_data/CIFAR', 'rb')
        data = pickle.load(f)
        f.close()

        x_train = []
        y_train = []

        for image in data['train']: #training already partitioned
            img = image['image']
            label = image['label']
            x_train.append(img)
            y_train.append(label)

        x_test = []
        y_test = []

        for image in data['test']: #extract testing data
            img = image['image']
            label = image['label']
            x_test.append(img)
            y_test.append(label)

        #make it a np array
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print("Training data:", x_train.shape)
        print("Testing data:", x_test.shape)

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }