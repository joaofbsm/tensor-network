#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Simple Neural Network example with Keras on top of Tensor Flow.
Python 2.7 was necessary because of an issue with Tensor Flow and Python 3.6.
"""

from __future__ import print_function

__author__ = "João Francisco Barreto da Silva Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "MIT"

import matplotlib.pyplot as plt
import numpy as np
import keras
import sys
import utils

from collections import Counter
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold


def train_network(X, y, train_index, test_index, parameters, extra=False):
    """Create and fit the MLP network
    
    Arguments:
        X -- Dataset input instances.
        y -- Dataset output classes.
        train_index -- K-fold generated train indexes.
        test_index -- K-fold generated test indexes.
        parameters -- Neural network model parameters.
    
    Keyword arguments:
        extra -- Flag for the presence of extra hidden layer (default: {False})
    """

    model = Sequential()  # Linear stack of layers
    sgd = SGD(lr=parameters["l_rate"])  # SGD optimizer. Allows learning rate.

    # Add hidden layer
    model.add(Dense(parameters["hidden_size"], activation="sigmoid",
                    input_dim=parameters["input_size"]))

    # Add extra layer if needed
    if extra:
        model.add(Dense(parameters["extra_size"], activation="sigmoid"))

    # Add output layer
    model.add(Dense(parameters["output_size"], activation="softmax"))

    # Split input by k-fold generated indexes
    train_X = X[train_index]
    test_X = X[test_index]

    # Convert output to one-hot encoding
    train_y = to_categorical(y[train_index], num_classes=7)
    test_y = to_categorical(y[test_index], num_classes=7)

    # Compile model
    model.compile(optimizer=sgd, loss="categorical_crossentropy", 
                  metrics=["accuracy"])

    # Fit model
    results = model.fit(x=train_X, y=train_y, 
                        batch_size=parameters["batch_size"],
                        epochs=parameters["n_epochs"], 
                        validation_data=(test_X, test_y), shuffle=True)

    return results.history


def execute_experiment(name, variations, X, y, parameters, kfold):
    """Train neural network for a set of different parameters and save results
    
    Arguments:
        name -- Name of the parameter to be varied.
        variations -- Variations of the parameter.
        X -- Dataset input instances.
        y -- Dataset output classes.
        parameters --  Neural network model parameters.
        kfold -- Object used to create k-folds for crossvalidation.
    """

    parameters = deepcopy(parameters)
    accuracy = {}
    
    for variation in variations:
        parameters[name] = variation
        accuracy_train = []  # Cross-validation train accuracy
        accuracy_test = []  # Cross-validation test accuracy
        
        for train_index, test_index in kfold.split(X, y):
            if name == "extra_sizes":
                results = train_network(X, y, train_index, test_index, 
                                        parameters, True)    
            else:
                results = train_network(X, y, train_index, test_index, 
                                        parameters)

            accuracy_train.append(results["acc"])
            accuracy_test.append(results["val_acc"])

        accuracy[variation] = {
            "train_mean": np.mean(accuracy_train, axis=0),
            "train_std": np.std(accuracy_train, axis=0),
            "test_mean": np.mean(accuracy_test, axis=0),
            "test_std": np.std(accuracy_test, axis=0)
        }

    utils.save_data(name, accuracy)


def balanced_experiment(X, y, parameters, kfold):
    """Oversample the dataset to analyze the performanced on a balanced one
    
    Arguments:
        X -- Dataset input instances.
        y -- Dataset output classes.
        parameters --  Neural network model parameters.
        kfold -- Object used to create k-folds for crossvalidation.
    """

    accuracy = {}
    
    # Oversample data
    X, y = RandomOverSampler().fit_sample(X, y)

    accuracy_train = []  # Cross-validation train accuracy
    accuracy_test = []  # Cross-validation test accuracy
    
    for train_index, test_index in kfold.split(X, y):
        results = train_network(X, y, train_index, test_index, parameters)
        accuracy_train.append(results["acc"])
        accuracy_test.append(results["val_acc"])

        accuracy[50] = {
            "train_mean": np.mean(accuracy_train, axis=0),
            "train_std": np.std(accuracy_train, axis=0),
            "test_mean": np.mean(accuracy_test, axis=0),
            "test_std": np.std(accuracy_test, axis=0)
        }

    utils.save_data("balanced", accuracy)


def main(args):
    X, y = utils.load_dataset(args[1])

    parameters = {
        "n_epochs": 200,  # Number of epochs
        "batch_size": 50,  # Default batch size
        "l_rate": 0.5,  # Default learning rate
        "input_size": 8,  # Input layer size
        "hidden_size": 50,  # Default hidden layer size
        "extra_size": 50,  # Default extra hidden layer size
        "output_size": 7  # Output layer size
    }  

    experiments = {
        "hidden_size": [10, 25, 50, 100],  # Hidden layer sizes
        "extra_size": [10, 25, 50, 100],  # Extra layer sizes
        "l_rate": [0.1, 0.5, 1, 10],  # Learning rates 
        "batch_size": [1, 10, 50, 100]  # Batch sizes
    }

    # Generator for 3-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True) 

    for key, value in experiments.items():
        print("\nExecuting {}\n----------------------\n".format(key))
        execute_experiment(key, value, X, y, parameters, kfold)

    balanced_experiment(X, y, parameters, kfold)

if __name__ == "__main__":
    main(sys.argv)