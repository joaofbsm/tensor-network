#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Utility functions"""

from __future__ import print_function

__author__ = "João Francisco Barreto da Silva Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "MIT"

import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path):
    """Load dataset from the specified file path.
    
    Arguments:
        file_path -- Path where the dataset is stored.
    """

    X = np.loadtxt(file_path, delimiter=',', usecols=tuple(range(8)))
    #y = LabelBinarizer().fit_transform(np.loadtxt(file_path, dtype='str', delimiter=',', usecols=8))
    y = LabelEncoder().fit_transform(np.loadtxt(file_path, dtype='str', delimiter=',', usecols=8))

    return X, y


def save_data(prefix, data):
    """Save data to respective files in results directory
    
    Arguments:
        prefix -- Experiment dependent prefix.
        data -- Data to be stored.
    """

    results_dir = "../results/{}".format(prefix)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for key, value in data.items():
        np.savetxt("{}/{}_train.csv".format(results_dir, key), value[0], delimiter=',')
        np.savetxt("{}/{}_test.csv".format(results_dir, key), value[1], delimiter=',')
