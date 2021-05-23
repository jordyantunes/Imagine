#!/usr/bin/env bash

python train.py \
--num_cpu=6 \
--policy_architecture=modular_attention \
--imagination_method=CGH \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=167 \
--admissible_attributes colors categories types sizes relative_sizes \
--furnitures door chair desk lamp \
--plants flower tree grass rose \
--animals dog cat human fly