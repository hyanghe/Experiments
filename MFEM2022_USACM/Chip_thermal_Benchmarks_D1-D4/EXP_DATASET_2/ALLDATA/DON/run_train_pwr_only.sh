#!/bin/bash

mode="train"
# variable="condition"
gpu=0
# load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True
load=True
# batch_size=20
learning_rate=3e-4
epochs=50000
# epochs=5

python ./DON.py -m ${mode} --gpu ${gpu} -lr ${learning_rate} -e ${epochs} -l ${load}

                                 
