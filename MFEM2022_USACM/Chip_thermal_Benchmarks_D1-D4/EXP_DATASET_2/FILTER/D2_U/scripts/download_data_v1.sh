#!/bin/bash


unzip train_hq.zip
mv train_hq/* data/imgs/
rm -d train_hq
rm train_hq.zip

unzip train_masks.zip
mv train_masks/* data/masks/
rm -d train_masks
rm train_masks.zip
