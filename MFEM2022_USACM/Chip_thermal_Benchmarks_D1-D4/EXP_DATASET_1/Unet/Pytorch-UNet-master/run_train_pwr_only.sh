#!/bin/bash

mode="train"
# variable="condition"
gpu=4
load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True

# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
python ./train_pwr_org_data.py -m ${mode} --gpu ${gpu} --load ${load}
# python ./train_pwr_org_data.py -m ${mode} --gpu ${gpu}

                                 
