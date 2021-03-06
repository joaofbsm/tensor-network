#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Plot neural network data for one vs two hidden layers"""

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

name = "extra_size"

n_epochs = 200
default_size = 50

# One layer

mean = np.loadtxt("{}/hidden_size/{}_train_mean.csv".format(results_dir, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="1 hidden layer training")

#mean = np.loadtxt("{}/hidden_size/{}_test_mean.csv".format(results_dir, default_size),
#                  delimiter=',')
#ax.plot(np.arange(n_epochs), mean, label="1 hidden layer testing")

# Two layers

mean = np.loadtxt("{}/{}/{}_train_mean.csv".format(results_dir, name, default_size),
                  delimiter=',')
ax.plot(np.arange(n_epochs), mean, label="2 hidden layers training")

#mean = np.loadtxt("{}/{}/{}_test_mean.csv".format(results_dir, name, default_size),
#                  delimiter=',')
#ax.plot(np.arange(n_epochs), mean, label="2 hidden layers testing")

ax.legend(loc="upper left")

#plt.savefig("{}/{}.png".format(plots_dir, name), dpi=300, bbox_inches="tight")

plt.show(block=False)
input("Hit Enter To Close")
plt.close()
