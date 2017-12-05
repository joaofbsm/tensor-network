#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Plot neural network data for identifying overfitting in training dataset"""

from __future__ import print_function

__author__ = "João Francisco Barreto da Silva Martins"
__email__ = "joaofbsm@dcc.ufmg.br"
__license__ = "MIT"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


plt.style.use("ggplot")

results_dir = "../../results"
plots_dir = "../../plots"

n_epochs = 200
default_size = 50

experiments = {
    "hidden_size": [5, 15, 50, 100],  # Hidden layer sizes
    "extra_size": [10, 25, 50, 100],  # Extra layer sizes
    "l_rate": [0.1, 0.5, 1, 10],  # Learning rates 
    "batch_size": [1, 10, 50, 100]  # Batch sizes
}

for key, values in experiments.items():
    for value in values:
        fig, ax = plt.subplots()

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")

        mean = np.loadtxt("{}/{}/{}_train_mean.csv".format(results_dir, key, value),
                          delimiter=',')
        std_dev = np.loadtxt("{}/{}/{}_train_std.csv".format(results_dir, key, value), delimiter=',')

        ax.plot(np.arange(n_epochs), mean, label="Training")
        ax.fill_between(np.arange(n_epochs), mean - std_dev, mean + std_dev, alpha=0.2)

        mean = np.loadtxt("{}/{}/{}_test_mean.csv".format(results_dir, key, value),
                          delimiter=',')
        std_dev = np.loadtxt("{}/{}/{}_test_std.csv".format(results_dir, key, value), delimiter=',')

        ax.plot(np.arange(n_epochs), mean, label="Testing")
        ax.fill_between(np.arange(n_epochs), mean - std_dev, mean + std_dev, alpha=0.2)

        ax.legend(loc="upper left")

        plt.savefig("{}/overfitting_{}_{}.png".format(plots_dir, key, value), dpi=300, bbox_inches="tight")

"""
plt.show(block=False)
input("Hit Enter To Close")
plt.close()
"""
