python train.py --num_cpu=6 --policy_architecture=modular_attention --imagination_method=CGH --reward_function=learned_lstm  --goal_invention=from_epoch_10 --n_epochs=167 --admissible_attributes colors categories types sizes relative_sizes --furnitures door chair desk lamp --plants flower tree bush grass rose --animals dog cat human fly mouse lion

python train.py --num_cpu=6 --policy_architecture=modular_attention --imagination_method=oracle --reward_function=learned_lstm  --goal_invention=from_epoch_10 --n_epochs=167 --admissible_attributes colors categories types sizes relative_sizes --furnitures door chair desk lamp --plants flower tree bush grass rose --animals dog cat human fly mouse lion

--custom_mpi_params "--oversubscribe"

docker run -it -d --name imagine -v noimagination:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name imagine -v cgh:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name imagine-oracle -v oracle:/home/localuser/src/data --privileged jordyantunes/imagine:latest

Proposta de Valor
Quem Somos Nós
Onde investimos
Cases
Nosso Resultados

python train.py \
--num_cpu=6 \
--policy_architecture=modular_attention \
--imagination_method=oracle \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=167 \
--admissible_attributes colors categories types sizes relative_sizes \
--furnitures door chair desk lamp \
--plants flower tree grass rose \
--animals dog cat human fly