#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Plot neural network data for varying hidden layer sizes"""

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
parameters = [5, 15, 50, 100]

for value in parameters:
    mean = np.loadtxt("{}/{}/{}_train_mean.csv".format(results_dir, name, value),
                      delimiter=',')
    
    ax.plot(np.arange(n_epochs), mean, label=str(value))


ax.legend(loc="upper left")

#plt.savefig("{}/{}.png".format(plots_dir, name), dpi=300, bbox_inches="tight")

plt.show(block=False)
input("Hit Enter To Close")
plt.close()
