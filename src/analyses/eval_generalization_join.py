import os
import json
import sys
sys.path.append('../../')
from src.utils.util import set_global_seeds
import src.imagine.experiment.config as config
from src.imagine.interaction import RolloutWorker
from src.imagine.goal_sampler import GoalSampler
from src.playground_env.reward_function import get_reward_from_state
from src.playground_env.env_params import get_env_params
from src.utils.util import get_stat_func
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gym
import argparse
import torch
import re
from itertools import chain

font = {'size'   : 25}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
                  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
                  [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]

PATH = 'path_to_folder_with_trial_ids_folders/'

RUN = True
PLOT = True
FREQ = 10
RENDER = 0
N_REPET = 1
LINE = 'mean'
ERR = 'sem'

test_set_def = None
types_words = None
type_legends = None
n_types = None

def init_test_set(params):
    global test_set_def, types_words, type_legends, n_types
    test_set_def = params['words_test_set_def']
    types_words = list(test_set_def.values())
    type_legends = ['Type {}'.format(i) for i in test_set_def.keys()]
    n_types = len(test_set_def.keys())

    return types_words, type_legends, n_types, test_set_def

def run_generalization_study(path, freq=10):
    first = True

    ignore_list = []
    ignore_path = os.path.join(path, '.ignore')
    if os.path.isfile(ignore_path):
        with open(ignore_path, 'r') as f:
            ignore_list = [i.strip() for i in f.readlines()]

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in ignore_list]

    for t_id, trial in enumerate(dirs):
        print(trial)
        t_init = time.time()
        trial_folder = path + '/' + trial + '/'
        policy_folder = trial_folder + 'policy_checkpoints/'
        params_file = trial_folder + 'params.json'

        data = pd.read_csv(os.path.join(trial_folder, 'progress.csv'))
        all_epochs = data['epoch']
        all_episodes = data['episode']
        epochs = []
        episodes = []
        for epoch, episode in zip(all_epochs, all_episodes):
            if epoch % freq == 0:
                epochs.append(epoch)
                episodes.append(int(episode))

        # Load params
        with open(params_file) as json_file:
            params = json.load(json_file)
        seed = params['experiment_params']['seed']
        set_global_seeds(seed)

        goal_invention = int(params['conditions']['goal_invention'].split('_')[-1])
        test_descriptions = params['test_descriptions']

        rank = 0
        if first:
            if not RENDER:
                env = 'PlaygroundNavigation-v1'
            else:
                env = 'PlaygroundNavigationRender-v1'
            params, rank_seed = config.configure_everything(rank=rank,
                                                            seed=seed,
                                                            num_cpu=params['experiment_params']['n_cpus'],
                                                            env=env,
                                                            trial_id=0,
                                                            n_epochs=10,
                                                            reward_function=params['conditions']['reward_function'],
                                                            policy_encoding=params['conditions']['policy_encoding'],
                                                            bias_buffer=params['conditions'].get('bias_buffer'),
                                                            feedback_strategy=params['conditions']['feedback_strategy'],
                                                            policy_architecture=params['conditions']['policy_architecture'],
                                                            goal_invention=params['conditions']['goal_invention'],
                                                            reward_checkpoint=params['conditions']['reward_checkpoint'],
                                                            rl_positive_ratio=params['conditions']['rl_positive_ratio'],
                                                            p_partner_availability=params['conditions']['p_social_partner_availability'],
                                                            git_commit='',
                                                            imagination_method=params['conditions']['imagination_method'],
                                                            admissible_attributes=params['env_params'].get('admissible_attributes'),
                                                            cuda=params['env_params'].get('cuda'),
                                                            **params['env_params'].get('categories', {}))

            policy_language_model, reward_language_model = config.get_language_models(params)

            onehot_encoder = config.get_one_hot_encoder(params['train_descriptions'] + params['test_descriptions'])
            goal_sampler = GoalSampler(policy_language_model=policy_language_model,
                                       reward_language_model=reward_language_model,
                                       goal_dim=policy_language_model.goal_dim,
                                       one_hot_encoder=onehot_encoder,
                                       **params.get('goal_sampler', {}),
                                       params=params)


            reward_function = config.get_reward_function(goal_sampler, params)
        else:
            def make_env():
                return gym.make(params['conditions']['env_name'])

            params['make_env'] = make_env
        
        loaded = False
        success_rates = np.zeros([len(test_descriptions), len(epochs)])
        if params['conditions']['reward_function'] == 'pretrained':
            reward_function.load_params(trial_folder + 'params_reward')
        if not loaded:
            # Load policy.
            t_init = time.time()

            for ind_ep, epoch in enumerate(epochs):
                print(time.time() - t_init)
                t_init = time.time()

                print('\n\n\t\t EPOCH', epoch)
                if first:
                    first = False
                    reuse = False
                else:
                    reuse = True

                if params['conditions']['reward_function'] == 'learned_lstm':
                    reward_function.restore_from_checkpoint(trial_folder + 'reward_checkpoints/reward_func_checkpoint_{}'.format(epoch))

                policy_language_model.set_reward_function(reward_function)
                if reward_language_model is not None:
                    reward_language_model.set_reward_function(reward_function)

                goal_sampler.update_discovered_goals(params['all_descriptions'], episode_count=0, epoch=0)

                with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=reuse):
                    policy = config.configure_learning_algo(reward_function=reward_function,
                                            goal_sampler=goal_sampler,
                                            params=params)
                    policy.load_params(policy_folder + 'policy_{}.pkl'.format(epoch))
                    # with open(policy_folder + 'policy_{}.pkl'.format(epoch), 'rb') as f:
                    #     policy = torch.load(f)

                evaluation_worker = RolloutWorker(make_env=params['make_env'],
                                                  policy=policy,
                                                  reward_function=reward_function,
                                                  params=params,
                                                  render=RENDER,
                                                  **params['evaluation_rollout_params'])
                evaluation_worker.seed(seed)

                # Run evaluation.
                evaluation_worker.clear_history()
                successes_per_descr = np.zeros([len(test_descriptions)])

                for ind_inst, instruction in enumerate(test_descriptions):
                    # instruction = 'Grasp any fly'
                    success_instruction = []
                    goal_str = [instruction]
                    goal_encoding = [policy_language_model.encode(goal_str[0])]
                    goal_id = [0]
                    for i in range(N_REPET):
                        ep = evaluation_worker.generate_rollouts(exploit=True,
                                                                 imagined=False,
                                                                 goals_str=goal_str,
                                                                 goals_encodings=goal_encoding,
                                                                 goals_ids=goal_id)
                        success = get_reward_from_state(state=ep[0]['obs'][-1], goal=instruction, params=params['env_params'])
                        success_instruction.append(success)
                    success_rate_inst = np.mean(success_instruction)
                    successes_per_descr[ind_inst] = success_rate_inst
                    print('\t Success rate {}: {}'.format(goal_str[0], success_rate_inst))
                    success_rates[ind_inst, ind_ep] = success_rate_inst
                np.savetxt(trial_folder + 'generalization_success_rates.txt', success_rates)

def plot_generalization(path, freq):
    ignore_list = []
    ignore_path = os.path.join(path, '.ignore')
    if os.path.isfile(ignore_path):
        with open(ignore_path, 'r') as f:
            ignore_list = [i.strip() for i in f.readlines()]

    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d not in ignore_list]
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    

    for i, trial in enumerate(dirs):
        print(trial)
        t_init = time.time()
        trial_folder = path + '/' + trial + '/'
        policy_folder = trial_folder + 'policy_checkpoints/'
        params_file = trial_folder + 'params.json'

        data = pd.read_csv(os.path.join(trial_folder, 'progress.csv'))
        all_epochs = data['epoch']
        all_episodes = data['episode']
        epochs = []
        episodes = []
        for epoch, episode in zip(all_epochs, all_episodes):
            if epoch % freq == 0:
                epochs.append(epoch)
                episodes.append(int(episode))

        # Load params
        with open(params_file) as json_file:
            params = json.load(json_file)

        seed = params['experiment_params']['seed']
        set_global_seeds(seed)

        goal_invention = int(params['conditions']['goal_invention'].split('_')[-1])
        test_descriptions = params['test_descriptions']

        success_rates = np.loadtxt(path + '/' + trial + '/generalization_success_rates.txt')

        to_plot = success_rates
        # TODO remove -------------------------------------------------------------
        # params = get_env_params(max_nb_objects=params['env_params'].get('max_nb_objects'),
        #             admissible_actions=params['env_params'].get('admissible_actions'),
        #             admissible_attributes=params['env_params'].get('admissible_attributes'),
        #             min_max_sizes=params['env_params'].get('min_max_sizes'),
        #             agent_size=params['env_params'].get('agent_size'),
        #             epsilon_initial_pos=params['env_params'].get('epsilon_initial_pos'),
        #             screen_size=params['env_params'].get('screen_size'),
        #             next_to_epsilon=params['env_params'].get('next_to_epsilon'),
        #             attribute_combinations=params['env_params'].get('attribute_combinations'),
        #             obj_size_update=params['env_params'].get('obj_size_update'),
        #             render_mode=params['env_params'].get('render_mode'),
        #             cuda=params['env_params'].get('cuda'),
        #             furnitures=params['env_params'].get('furnitures'),
        #             plants=params['env_params'].get('plants'),
        #             animals=params['env_params'].get('animals'),
        #             supplies=params['env_params'].get('supplies'))
        # _, type_legends, _, test_set_def = init_test_set(params)

        # test_descriptions_df = pd.Series(test_descriptions)
        # test_type_mapping = {}

        # for type_index, type_descriptions in test_set_def.items():
        #     inds_desc = []
        #     for i_test_d, test_descr in test_descriptions_df.items():
        #         for type_d in type_descriptions:
        #             to_add = None
        #             if isinstance(type_d, re.Pattern):
        #                 r = type_d.search(test_descr)
        #                 if r is not None:
        #                     to_add = i_test_d
        #             elif type_d in test_descr:
        #                 to_add = i_test_d
        #             if not type_index in (2, 3, 4, 5):
        #                 if type_index == 1 and re.match('Grow.+door', test_descr):
        #                     print("Removing unachievable", test_descr)
        #                 else:
        #                     continue
        #             if to_add is not None:
        #                 print("Removing Type", type_index,test_descr)
        #                 inds_desc.append(to_add)
        #                 break
        #     test_type_mapping[type_index] = inds_desc
        
        # to_remove_ids = list(set(chain.from_iterable(test_type_mapping.values())))
        # test_descriptions_df = test_descriptions_df.drop(index=to_remove_ids)

        # print("Original test_descriptions", len(test_descriptions), "after removal", test_descriptions_df.shape[0])
        # print("To plot", to_plot.shape)
        # to_plot = to_plot[np.array(test_descriptions_df.index),:]
        # print("To plot", to_plot.shape)
        # TODO remove -------------------------------------------------------------

        line, err_min, err_max = get_stat_func(LINE, ERR)
        first = False
        # plot
        plt.plot(np.array(episodes) / 1000, line(to_plot), linewidth=10, c=colors[i])
        plt.fill_between(np.array(episodes) / 1000, err_min(to_plot), err_max(to_plot), color=colors[i], alpha=0.2)
    leg = plt.legend(dirs, frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Average success rate')
    plt.savefig(os.path.join(path, 'all_generalization_test_set_policy.pdf'), bbox_extra_artists=(lab, lab2), bbox_inches='tight',
                    dpi=50)  # add leg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--path', type=str, default=PATH)
    add('--plot', default=PLOT, type=lambda x: (str(x).lower() == 'true'))
    add('--run', default=RUN, type=lambda x: (str(x).lower() == 'true'))
    kwargs = vars(parser.parse_args())
    if kwargs['run']:
        run_generalization_study(kwargs['path'], FREQ)
    if kwargs['plot']:
        plot_generalization(kwargs['path'], FREQ)