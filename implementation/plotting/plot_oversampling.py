#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Plot neural network data to compare under and over sampled datasets"""

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

name = "balanced"

n_epochs = 200
default_size = 15

# Undersampled

mean = np.loadtxt("{}/hidden_size/{}_train_mean.csv".format(results_dir, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="Undersampled training")

mean = np.loadtxt("{}/hidden_size/{}_test_mean.csv".format(results_dir, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="Undersampled testing")

# Oversampled

mean = np.loadtxt("{}/{}/{}_train_mean.csv".format(results_dir, name, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="Oversampled training")

mean = np.loadtxt("{}/{}/{}_test_mean.csv".format(results_dir, name, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="Oversampled testing")

ax.legend(loc="upper left")

#plt.savefig("{}/oversampling.png".format(plots_dir), dpi=300, bbox_inches="tight")

plt.show(block=False)
input("Hit Enter To Close")
plt.close()
