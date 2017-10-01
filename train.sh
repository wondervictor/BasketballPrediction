#!/usr/bin/env bash

# generate training set and testing set

python data_progresss.py

# train

python train.py --model dnn --epoch 60 --team_data_type reduce --cuda 1 --batch-size 16