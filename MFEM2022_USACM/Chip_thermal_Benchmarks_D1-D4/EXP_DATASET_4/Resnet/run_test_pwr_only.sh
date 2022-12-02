#!/bin/bash

mode="test"
# variable="condition"
gpu=2
# load='./checkpoints/checkpoint.pth'
# load = True
# plot_solution=True
# batch_size=20
learning_rate=1e-5
# epochs=50000

python ./main_v01.py -m ${mode} --gpu ${gpu} -lr ${learning_rate}

                                 
