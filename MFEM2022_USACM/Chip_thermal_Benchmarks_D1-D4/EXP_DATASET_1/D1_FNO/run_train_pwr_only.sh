#!/bin/bash

mode="train"
# variable="condition"
gpu=2
# load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True
batch_size=20
learning_rate=0.001
epochs=50000

python ./fourier_2d_pwr_valLoader.py -m ${mode} --gpu ${gpu} -b ${batch_size} -l ${learning_rate} -e ${epochs}

                                 
