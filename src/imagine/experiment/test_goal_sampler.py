import numpy as np
import sys

sys.path.append('../../../')

from src.imagine.goal_generator.simple_sentence_generator import SentenceGeneratorHeuristic
from src.playground_env.descriptions import generate_all_descriptions
from src.playground_env.env_params import get_env_params, init_params
from src.imagine.experiment import config
from src.imagine.goal_sampler import GoalSampler, EvalGoalSampler
from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state

input_params = {
    "admissible_attributes": ['colors', 'categories', 'types', 'sizes', 'relative_sizes'],
    "furnitures": ['door', 'chair', 'desk', 'lamp'],
    "plants": ['flower', 'tree', 'bush', 'rose'],
    "animals": ['dog', 'cat', 'human', 'fly']
}

init_params(**input_params)
params = get_env_params(render_mode=True)
train, test, extra = generate_all_descriptions(params)
train, test = sorted(train), sorted(test)

print("train descriptions", len(train), "test descriptions", len(test))

for m in ['CGH', 'oracle']:
    print("-------------------------------------------------------------------------------------")
    print(m)
    generator = SentenceGeneratorHeuristic(train_descriptions=train,
                                        test_descriptions=test,
                                        sentences=None, 
                                        method=m)

    # update the set of known goals
    generator.update_model(train)
    # generate imagined goals
    new_descriptions = generator.generate_sentences(epoch=20)

    print(sorted((set(test) - set(new_descriptions)) - set(generator.descriptions_impossible_to_imagine)))
    # generator.update_model(new_descriptions)
    # print('---------------------------------segunda vez---------------------------------')
    # new_descriptions = generator.generate_sentences()

    p_found_in_test = sum([d in test for d in new_descriptions]) / len(test)
    p_not_in_test = sum([d not in test for d in new_descriptions]) / len(new_descriptions)
    p_in_test = sum([d in test for d in new_descriptions]) / len(new_descriptions)
    print('Percentage of the test set found:', p_found_in_test)
    print('Percentage of the new descriptions that are not in the test', p_not_in_test)
    print('Percentage of the new descriptions that are in the test set', p_in_test)
    del generator
# oracle
# Percentage of the test set found: 0.7122302158273381
# Percentage of the new descriptions that are not in the test 0.0
# Percentage of the new descriptions that are in the test set 1.0

# cgh
# Percentage of the test set found: 0.7122302158273381
# Percentage of the new descriptions that are not in the test 0.6796116504854369 
# Percentage of the new descriptions that are in the test set 0.32038834951456313

print('-')