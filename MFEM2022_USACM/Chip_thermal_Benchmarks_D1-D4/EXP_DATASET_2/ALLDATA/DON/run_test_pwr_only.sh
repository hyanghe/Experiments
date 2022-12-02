#!/bin/bash

mode="test"
# variable="condition"
gpu=0
# load='./checkpoints/checkpoint.pth'
# load = True
# plot_solution=True
# batch_size=20
learning_rate=1e-4
# epochs=50000

python ./DON.py -m ${mode} --gpu ${gpu} -lr ${learning_rate}

                                 
