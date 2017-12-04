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
        X -- Dataset input instances
        y -- Dataset output classes
        train_index -- K-fold generated train indexes
        test_index -- K-fold generated test indexes
        parameters -- Neural network model parameters
    
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


def hidden_analysis(X, y, parameters, hidden_sizes, kfold):
    """Analyze the perfomance of the network on varying hidden layer sizes."""

    parameters = deepcopy(parameters)
    accuracy = {}
    
    for hidden_size in hidden_sizes:
        parameters["hidden_size"] = hidden_size
        cv_accuracy = []  # Cross-validation accuracy

        for train_index, test_index in kfold.split(X, y):
            results = train_network(X, y, train_index, test_index, parameters)
            cv_accuracy.append(results["acc"])

        accuracy[hidden_size] = np.mean(cv_accuracy, axis=0)

    utils.save_data("hidden", accuracy)


def extra_analysis(X, y, parameters, extra_sizes, kfold):
    """Analyze the perfomance of the network with one extra hidden layer."""
    
    parameters = deepcopy(parameters)
    accuracy = {}
    
    for extra_size in extra_sizes:
        parameters["extra_size"] = extra_size
        cv_accuracy = []  # Cross-validation accuracy
        
        for train_index, test_index in kfold.split(X, y):
            results = train_network(X, y, train_index, test_index, parameters, 
                                    extra=True)
            cv_accuracy.append(results["acc"])

        accuracy[extra_size] = np.mean(cv_accuracy, axis=0)

    utils.save_data("extra", accuracy)


def learning_analysis(X, y, parameters, l_rates, kfold):
    """Analyze the perfomance of the network on varying learning rates."""
    
    parameters = deepcopy(parameters)
    accuracy = {}
    
    for l_rate in l_rates:
        parameters["l_rate"] = l_rate
        cv_accuracy = []  # Cross-validation accuracy
        
        for train_index, test_index in kfold.split(X, y):
            results = train_network(X, y, train_index, test_index, parameters)
            cv_accuracy.append(results["acc"])

        accuracy[l_rate] = np.mean(cv_accuracy, axis=0)

    utils.save_data("l_rate", accuracy)


def batch_analysis(X, y, parameters, batch_sizes, kfold):
    """Analyze the perfomance of the network on varying batch sizes."""
    
    parameters = deepcopy(parameters)
    accuracy = {}
    
    for batch_size in batch_sizes:
        parameters["batch_size"] = batch_size
        cv_accuracy = []  # Cross-validation accuracy
        
        for train_index, test_index in kfold.split(X, y):
            results = train_network(X, y, train_index, test_index, parameters)
            cv_accuracy.append(results["acc"])

        accuracy[batch_size] = np.mean(cv_accuracy, axis=0)

    utils.save_data("batch_size", accuracy)


def balanced_analysis():
    """Analyze the perfomance of the network on a balanced dataset."""
    
    pass


def main(args):
    X, y = utils.load_dataset(args[1])

    parameters = {
        "n_epochs": 150,  # Number of epochs
        "batch_size": 100,  # Default batch size
        "l_rate": 1,  # Default learning rate
        "input_size": 8,  # Input layer size
        "hidden_size": 50,  # Default hidden layer size
        "extra_size": 50,  # Default extra hidden layer size
        "output_size": 7,  # Output layer size
    }  

    # Variable parameters
    batch_sizes = [1, 10, 50, 100]  # Batch sizes
    l_rates = [0.1, 0.5, 1, 10]  # Learning rates 
    hidden_sizes = [10, 25, 50, 100]  # Hidden layer sizes
    extra_sizes = [10, 25, 50, 100]  # Extra layer sizes


    # Generator for 3-fold cross-validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True) 

    hidden_analysis(X, y, parameters, hidden_sizes, kfold)

    extra_analysis(X, y, parameters, extra_sizes, kfold)

    learning_analysis(X, y, parameters, l_rates, kfold)

    batch_analysis(X, y, parameters, batch_sizes, kfold)



if __name__ == "__main__":
    main(sys.argv)