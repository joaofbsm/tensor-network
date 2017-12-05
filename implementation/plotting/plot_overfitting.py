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

fig, ax = plt.subplots()

ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")

results_dir = "../../results"
plots_dir = "../../plots"

name = "hidden_size"

n_epochs = 200
default_size = 50

mean = np.loadtxt("{}/{}/{}_train_mean.csv".format(results_dir, name, default_size),
                  delimiter=',')
std_dev = np.loadtxt("{}/{}/{}_train_std.csv".format(results_dir, name, 
                     default_size), delimiter=',')

ax.plot(np.arange(n_epochs), mean, label="Training")
ax.fill_between(np.arange(n_epochs), mean - std_dev, mean + std_dev, alpha=0.2)

mean = np.loadtxt("{}/{}/{}_test_mean.csv".format(results_dir, name, default_size),
                  delimiter=',')
std_dev = np.loadtxt("{}/{}/{}_test_std.csv".format(results_dir, name, 
                     default_size), delimiter=',')

ax.plot(np.arange(n_epochs), mean, label="Testing")
ax.fill_between(np.arange(n_epochs), mean - std_dev, mean + std_dev, alpha=0.2)

ax.legend(loc="upper left")

#plt.savefig("{}/overfitting.png".format(plots_dir), dpi=300, bbox_inches="tight")

plt.show(block=False)
input("Hit Enter To Close")
plt.close()
