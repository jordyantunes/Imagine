python train-compound.py `
--num_cpu=1 `
--policy_architecture=modular_attention `
--imagination_method=CGH `
--reward_function=learned_lstm  `
--goal_invention=from_epoch_10 `
--n_epochs=167 `
--admissible_actions Go Grasp Grow Turn Pour `
--admissible_attributes colors categories types status `
--max-nb-objects 4
