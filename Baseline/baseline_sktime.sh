#!/bin/bash

#SBATCH --gres=gpu:1 --cpus-per-task=8 --mail-type=ALL

. $HOME/anaconda3/etc/profile.d/conda.sh

conda activate sktime

ipython ./baseline_sktime.py
