#!/bin/bash

#SBATCH --gres=gpu:1 --cpus-per-task=4 --mail-type=ALL

. $HOME/anaconda3/etc/profile.d/conda.sh

ipython ./baseline_pytorch_500.py
