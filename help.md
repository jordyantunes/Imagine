python train.py --num_cpu=6 --policy_architecture=modular_attention --imagination_method=CGH --reward_function=learned_lstm  --goal_invention=from_epoch_10 --n_epochs=167 --admissible_attributes colors categories types sizes relative_sizes --furnitures door chair desk lamp --plants flower tree bush grass rose --animals dog cat human fly mouse lion

python train.py --num_cpu=6 --policy_architecture=modular_attention --imagination_method=oracle --reward_function=learned_lstm  --goal_invention=from_epoch_10 --n_epochs=167 --admissible_attributes colors categories types sizes relative_sizes --furnitures door chair desk lamp --plants flower tree bush grass rose --animals dog cat human fly mouse lion

--custom_mpi_params "--oversubscribe"

docker run -it -d --name imagine -v noimagination:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name imagine -v cgh:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name imagine-oracle -v oracle:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name imagine -v experiment:/home/localuser/src/data --privileged jordyantunes/imagine:latest

docker run -it -d --name compound -v compound:/home/localuser/src/data --privileged jordyantunes/imagine:compound

docker run -it -d --name compound-new -v compound-new:/home/localuser/src/data --privileged jordyantunes/imagine:compound

docker run -it -d --name compound-mem -v compound-mem:/home/localuser/src/data --privileged jordyantunes/imagine:compound

docker run -it -d --name compound-reduced -v compound-reduced:/home/localuser/src/data --privileged jordyantunes/imagine:compound-reduced

docker run -it -d --name no-compound -v no-compound:/home/localuser/src/data --privileged jordyantunes/imagine:compound

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

# compount
nohup \
python train-compound.py \
--num_cpu=8 \
--policy_architecture=modular_attention \
--imagination_method=CGH \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=201 \
--admissible_attributes colors categories types sizes relative_sizes \
--admissible_actions Move Grasp Grow Turn Pour \
--compound_goals_from=120 \
--admissible_actions Move Grasp Grow Turn Pour \
--admissible_attributes colors categories types status under_lighting \
--max-nb-objects 6 \
--add-light &

# compound reduced
nohup \
python train-compound.py \
--num_cpu=8 \
--policy_architecture=modular_attention \
--imagination_method=CGH \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=201 \
--admissible_attributes colors categories types sizes relative_sizes \
--admissible_actions Move Grasp Grow Turn Pour \
--compound_goals_from=120 \
--admissible_actions Move Grasp Grow Turn Pour \
--admissible_attributes colors categories types status under_lighting \
--max-nb-objects 6 \
--add-light \
--furnitures door chair desk lamp window \
--animals dog cat human fly \
--compound-config category_only:true &

# no compound
nohup \
python train-compound.py \
--num_cpu=8 \
--policy_architecture=modular_attention \
--imagination_method=CGH \
--reward_function=learned_lstm  \
--goal_invention=from_epoch_10 \
--n_epochs=201 \
--admissible_attributes colors categories types sizes relative_sizes \
--admissible_actions Move Grasp Grow Turn Pour \
--admissible_actions Move Grasp Grow Turn Pour \
--admissible_attributes colors categories types status under_lighting \
--max-nb-objects 6 \
--add-light \
--furnitures door chair desk lamp window \
--animals dog cat human fly &