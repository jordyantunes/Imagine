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
from src.imagine.goal_sampler import GoalSampler
import src.imagine.experiment.config as config
from src.playground_env.env_controller import EnvController

admissible_attributes=['colors', 'categories', 'types', 'under_lighting', 'status'] # , 'relative_sizes']

# furnitures = ['door', 'chair', 'desk', 'lamp']
# plants = ['flower', 'tree', 'bush', 'rose']
# animals = ['dog', 'cat', 'human', 'fly']

init_params(
            max_nb_objects = 4,
            admissible_actions = ('Move', 'Grasp', 'Grow', 'Turn', 'Pour'),
            admissible_attributes=admissible_attributes,
            # furnitures=furnitures,
            # plants=plants,
            # animals=animals
            )
params = get_env_params(render_mode=args.render)

# add conditions
params["conditions"] = {
    "imagination_method": "CGH",
    "env_name": "PlaygroundNavigation-v1",
    "env_id": "big",
    "policy_architecture": "modular_attention",
    "policy_encoding": "lstm",
    "reward_checkpoint": "",
    "goal_invention": "from_epoch_11",
    "p_imagined": 0.5,
    "curriculum_target": False,
    "curriculum_goal_discovery": True,
    "bias_buffer": True,
    "goal_sampling_policy": "random",
    "reward_function": "learned_lstm",
    "feedback_strategy": "exhaustive",
    "p_social_partner_availability": 1,
    "rl_positive_ratio": 0.5,
    "reward_positive_ratio": 0.2,
    "curriculum_replay_target": "no"
  }

params['dims'] = {
    "obs": 240,
    "g_encoding": 100,
    "g_id": 1,
    "acts": 4,
    "g_str": None,
    "nb_obj": 4,
    "inds_objs": [
      [
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41
      ],
      [
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80
      ],
      [
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119
      ]
    ],
    "inds_grasped_obj": []
  }

train, test, extra, train_descriptions_compound, test_descriptions_compound = generate_all_descriptions(params)

params["experiment_params"] = {
    "n_cpus": 6,
    "rollout_batch_size": 2,
    "n_cycles": 50,
}
params['train_descriptions'] = train
params['test_descriptions'] = test
params['extra_descriptions'] = extra
params['all_descriptions'] = train + test + extra
params['reward_function'] = {
    "batch_size": 512,
    "max_n_epoch": 100,
    "early_stopping": "f1",
    "n_batch": 200,
    "freq_update": 2,
    "freq_reset_update": 10,
    "n_epochs_warming_up": 500,
    "n_objs": 3,
    "learning_rate": 0.001,
    "ff_size": 100,
    "num_hidden_lstm": 100,
    "reward_positive_ratio": 0.2
  }
params["or_params_path"] = {
    2: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_2objs.pk",
    3: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_3objs.pk",
    4: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_4objs.pk",
    5: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_5objs.pk",
    6: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_6objs.pk",
    7: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_7objs.pk",
    8: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_8objs.pk",
    9: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_9objs.pk",
    10: "C:/Users/jordy/Documents/GitHub/Imagine/src/data/or_function/or_params_10objs.pk"
  }

if not params['render_mode']:
    env_name = 'PlaygroundNavigation-v1'
else:
    env_name = 'PlaygroundNavigationHuman-v1'

env = gym.make(env_name,
               display=params["render_mode"],
               admissible_attributes=params["admissible_attributes"])

env.reset()
# env.unwrapped.reset_with_goal("Grow red plant")
env.unwrapped.reset_with_goal("Random goal")

policy_language_model, reward_language_model = config.get_language_models(params)
onehot_encoder = config.get_one_hot_encoder(params['all_descriptions'])

goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                               reward_language_model=reward_language_model,
                               goal_dim=policy_language_model.goal_dim,
                               one_hot_encoder=onehot_encoder,
                               params=params)

reward_function = config.get_reward_function(goal_sampler, params)
policy_language_model.set_reward_function(reward_function)

stop = False
while not stop:
    # init_render

    action = np.zeros([4])
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
            # J4
            elif event.key == K_p:
                action[3] = 1

            elif event.key == K_q:
                stop = True

            if action.sum() != 0:
                time.sleep(0.05)
                break
        elif event.type == QUIT:
            stop = True

    out = env.step(action)
    env.render()

    # print(env.used_supplies)
    # Sample descriptions of the current state
    train_descr, test_descr, extra_descr = sample_descriptions_from_state(out[0], env.unwrapped.params)
    descr = train_descr + test_descr
    print(descr)

    for d in descr:
        encoded = policy_language_model.encode(d)

    # # assert that the reward function works, should give positive rewards for descriptions sampled, negative for others.
    # for d in descr:
    #     assert get_reward_from_state(out[0], d, env_params)
    # for d in np.random.choice(list(set(all_descriptions) - set(descr)), size=20):
    #     assert not get_reward_from_state(out[0], d, env_params)
else:
    pygame.quit()
    sys.exit()
