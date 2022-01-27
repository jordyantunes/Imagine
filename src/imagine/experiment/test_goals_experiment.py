import numpy as np
from pygame.locals import *
import pygame
import sys
import gym
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=False)

args = parser.parse_args()

sys.path.append('../../../')
from src.playground_env.descriptions import generate_all_descriptions
from src.playground_env.env_params import get_env_params, init_params
from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state
from src.playground_env.env_controller import EnvController

# admissible_attributes=['colors', 'categories', 'types', 'sizes', 'relative_sizes']

# furnitures = ['door', 'chair', 'desk', 'lamp']
# plants = ['flower', 'tree', 'bush', 'rose']
# animals = ['dog', 'cat', 'human', 'fly']

init_params(
            max_nb_objects = 4,
            admissible_actions = ('Move', 'Grasp', 'Grow', 'Turn')
            # admissible_attributes=admissible_attributes,
            # furnitures=furnitures,
            # plants=plants,
            # animals=animals
            )
params = get_env_params(render_mode=args.render)
train, test, extra = generate_all_descriptions(params)


if not params['render_mode']:
    env_name = 'PlaygroundNavigation-v1'
else:
    env_name = 'PlaygroundNavigationHuman-v1'

env = gym.make(env_name,
               display=params["render_mode"],
               admissible_attributes=params["admissible_attributes"])

env.reset()
env.unwrapped.reset_with_goal("Grow red plant")

stop = False
while not stop:
    # init_render

    action = np.zeros([3])
    for event in pygame.event.get():
        if hasattr(event, 'key'):
            # J1
            if (event.key == K_DOWN):
                action[1] = -1
            elif event.key == K_UP:
                action[1] = 1
            # J2
            elif (event.key == K_LEFT):
                action[0] = -1
            elif event.key == K_RIGHT:
                action[0] = 1
            # J3
            elif event.key == K_SPACE:
                action[2] = 1

            elif event.key == K_q:
                stop = True

            if action.sum() != 0:
                time.sleep(0.05)
                break
        elif event.type == QUIT:
            stop = True

    out = env.step(action)
    env.render()

    print(env.used_supplies)
    # Sample descriptions of the current state
    train_descr, test_descr, extra_descr = sample_descriptions_from_state(out[0], env.unwrapped.params)
    descr = train_descr + test_descr
    print(descr)

    # # assert that the reward function works, should give positive rewards for descriptions sampled, negative for others.
    # for d in descr:
    #     assert get_reward_from_state(out[0], d, env_params)
    # for d in np.random.choice(list(set(all_descriptions) - set(descr)), size=20):
    #     assert not get_reward_from_state(out[0], d, env_params)
else:
    pygame.quit()
    sys.exit()
