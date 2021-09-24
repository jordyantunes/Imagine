import numpy as np
from pygame.locals import *
import pygame
import sys
import gym
import time
import re
from itertools import chain

sys.path.append('../../../')
from src.playground_env.descriptions import generate_all_descriptions
from src.playground_env.env_params import get_env_params, init_params
from src.playground_env.reward_function import sample_descriptions_from_state, get_reward_from_state

# admissible_attributes=['colors', 'categories', 'types']
input_params = {
    "admissible_attributes": ['colors', 'categories', 'types', 'sizes', 'relative_sizes'],
    "furnitures": ['door', 'chair', 'desk', 'lamp'],
    "plants": ['flower', 'tree', 'bush', 'rose'],
    "animals": ['dog', 'cat', 'human', 'fly']
}

init_params(**input_params)
params = get_env_params(render_mode=True)
train, test, extra = generate_all_descriptions(params)

grow = [t.split(' ') for t in train if t.startswith("Grow")]
grow = list(set([g[0] + ' ' + g[-1] for g in grow]))

print("Types of grow")
print("Grow", grow)

wrong = [
    *[re.compile(f'Grow.+{w}') for w in params['categories']['furniture'] + params['categories']['supply'] + ('furniture', 'supply')],
    re.compile('Grasp.+animal'),
    re.compile('Grasp.+fly'),
    re.compile('Grow.+plants'),
    *[re.compile(f'Grow.+{w}') for w in params['categories']['plant'] + ('plant', 'living_thing')]
]

print("Wrong goals")
print([t for t in train for w in wrong if w.match(t)])

print("length = 2 goals")
print([g for g in train if len(g.split(' ')) == 2])

print("Train", len(train), "test", len(test))

words_in_train = set(chain.from_iterable([t.split(' ') for t in train]))
words_in_test = set(chain.from_iterable([t.split(' ') for t in test]))

print("Words only in test")
print(words_in_test - words_in_train)

exit(0)

if not params['render_mode']:
    env_name = 'PlaygroundNavigation-v1'
else:
    env_name = 'PlaygroundNavigationHuman-v1'

env = gym.make(env_name,
               display=params["render_mode"],
               admissible_attributes=params["admissible_attributes"])

env.reset()
env.unwrapped.reset_with_goal("Grow biggest red tree")

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
