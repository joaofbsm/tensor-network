#!/bin/bash
#title          :plot_all.sh
#description    :Plot all the graphics for neural network
#author         :Joao Francisco B. S. Martins
#date           :04.12.2017
#usage          :bash plot_all.sh
#bash_version   :GNU bash, version 4.4.0(1)-release
#==============================================================================

python plot_hidden.py
python plot_extra.py
python plot_l_rate.py
python plot_batch.py
python plot_overfitting.py
python plot_oversampling.py