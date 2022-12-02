#!/bin/bash

mode="test"
# variable="condition"
gpu=5
load='./checkpoints/checkpoint.pth'
# load = False
# plot_solution=True

# python ./generator_train_sln.py -m ${mode} --variable ${variable} --gpu ${gpu} --load_model ${load_model} --plot_solution ${plot_solution}
python ./train_pwr_param_Scale_Parall_v01.py -m ${mode} --gpu ${gpu} --load ${load}
# python ./train_pwr_v1_shift_Tmin.py -m ${mode} --gpu ${gpu}

                                 
