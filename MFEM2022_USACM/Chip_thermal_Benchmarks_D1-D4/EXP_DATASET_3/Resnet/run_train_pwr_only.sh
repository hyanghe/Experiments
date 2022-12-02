#!/bin/bash

mode="train"
# variable="condition"
gpu=3
# load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True
load=True
# batch_size=20
learning_rate=1e-5
epochs=50000
# epochs=5

python ./main_v01.py -m ${mode} --gpu ${gpu} -lr ${learning_rate} -e ${epochs} -l ${load}

                                 
