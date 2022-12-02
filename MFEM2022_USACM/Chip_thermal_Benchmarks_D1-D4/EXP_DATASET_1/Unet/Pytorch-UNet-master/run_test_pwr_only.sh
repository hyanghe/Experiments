#!/bin/bash

mode="test"
# variable="condition"
gpu=0
load='./checkpoints/checkpoint.pth'
epochs=50000
# load = False
# plot_solution=True

# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
python ./train_pwr_org_data.py -m ${mode} --gpu ${gpu} --load ${load} --epochs ${epochs}
# python ./train_pwr_v1_shift_Tmin.py -m ${mode} --gpu ${gpu}

                                 
