#!/usr/bin/env bash

python train-compound.py \
--num_cpu=6 \
--policy_architecture=modular_attention \
--imagination_method=CGH \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=200 \
--admissible_attributes colors categories types sizes relative_sizes \
--admissible_actions Go Grasp Grow Turn Pour \
--compound_goals_from=120 \
--admissible_actions Go Grasp Grow Turn Pour \
--admissible_attributes colors categories types status under_lighting \
--max-nb-objects 6 \
--add-light