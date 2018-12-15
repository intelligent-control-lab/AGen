import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import tensorflow as tf
import time

backend ='TkAgg'
import matplotlib
matplotlib.use(backend)
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from contexttimer import Timer

import hgail.misc.simulation
import hgail.misc.utils

import hyperparams
import utils
from utils import str2bool
import rls, pdb
import validate_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='../../data/experiments/multiagent_curr/')
    parser.add_argument('--params_filename', type=str, default='itr_200.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=10)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=True)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)

    run_args = parser.parse_args()

    args_filepath = os.path.join(run_args.exp_dir, 'imitate/log/args.npz')
    args = hyperparams.load_args(args_filepath)

    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    filenames = [
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]
 
    # batch_size=100, critic_batch_size=1000, critic_dropout_keep_prob=0.8, critic_grad_rescale=40.0, critic_hidden_layer_dims=(128, 128, 64), critic_learning_rate=0.0004, decay_reward=False, discount=0.99, do_curriculum=False, env_H=200, env_action_repeat=1, env_multiagent=True, env_primesteps=50, env_reward=0, exp_dir='../../data/experiments', exp_name='multiagent_curr_1_{}', expert_filepath='../../data/trajectories/ngsim.h5', gradient_penalty=2.0, itrs_per_decay=25, latent_dim=4, load_params_init='NONE', max_path_length=1000, n_critic_train_epochs=40, n_envs=1, n_envs_end=50, n_envs_start=10, n_envs_step=10, n_itr=2000, n_recognition_train_epochs=30, ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt', normalize_clip_std_multiple=10.0, params_filepath='', policy_mean_hidden_layer_dims=(128, 128, 64), policy_recurrent=True, policy_std_hidden_layer_dims=(128, 64), recognition_hidden_layer_dims=(128, 64), recognition_learning_rate=0.0005, recurrent_hidden_dim=64, remove_ngsim_veh=False, remove_ngsim_vehicles=False, render_every=25, reward_handler_critic_final_scale=1.0, reward_handler_max_epochs=100, reward_handler_recognition_final_scale=0.2, reward_handler_use_env_rewards=True, scheduler_k=20, trpo_step_size=0.01, use_critic_replay_memory=True, use_infogail=False, validator_render=False, vectorize=True)

    # (batch_size=100, critic_batch_size=1000, critic_dropout_keep_prob=0.8, critic_grad_rescale=40.0, critic_hidden_layer_dims=(128, 128, 64), critic_learning_rate=0.0004, decay_reward=False, discount=0.99, do_curriculum=False, env_H=200, env_action_repeat=1, env_multiagent=True, env_primesteps=50, env_reward=0, exp_dir='../../data/experiments', exp_name='multiagent_curr_1_{}', expert_filepath='../../data/trajectories/ngsim.h5', gradient_penalty=2.0, itrs_per_decay=25, latent_dim=4, load_params_init='NONE', max_path_length=1000, n_critic_train_epochs=40, n_envs=22, n_envs_end=50, n_envs_start=10, n_envs_step=10, n_itr=2000, n_recognition_train_epochs=30, ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt', normalize_clip_std_multiple=10.0, params_filepath='', policy_mean_hidden_layer_dims=(128, 128, 64), policy_recurrent=True, policy_std_hidden_layer_dims=(128, 64), recognition_hidden_layer_dims=(128, 64), recognition_learning_rate=0.0005, recurrent_hidden_dim=64, remove_ngsim_veh=False, remove_ngsim_vehicles=False, render_every=25, reward_handler_critic_final_scale=1.0, reward_handler_max_epochs=100, reward_handler_recognition_final_scale=0.2, reward_handler_use_env_rewards=True, scheduler_k=20, trpo_step_size=0.01, use_critic_replay_memory=True, use_infogail=False, validator_render=False, vectorize=True)

    if run_args.n_envs:
        args.n_envs = run_args.n_envs
    args.env_H = 200
    sys.stdout.write('{} vehicles with H = {}'.format(args.n_envs, args.env_H))
            
    for fn in filenames:
        args.ngsim_filename = fn
        if args.env_multiagent:
            # args.n_envs gives the number of simultaneous vehicles 
            # so run_args.n_multiagent_trajs / args.n_envs gives the number 
            # of simulations to run overall
            egoids = list(range(int(run_args.n_multiagent_trajs / args.n_envs)))
            starts = dict()
        else:
            egoids, starts = load_egoids(fn, args, run_args.n_runs_per_ego_id)


        params_filepath = os.path.join(run_args.exp_dir, 'imitate/log/{}'.format(run_args.params_filename))
        params = hgail.misc.utils.load_params(params_filepath)
        policy_fn = utils.build_hierarchy if run_args.use_hgail else utils.build_policy
        
        env, _, _ = utils.build_ngsim_env(args, alpha=0.)

        # policy = policy_fn(args, env)
        summary_writer = tf.summary.FileWriter(os.path.join('multiagent_curr', 'imitate', 'summaries'))
        data = validate_utils.get_ground_truth()
        
        critic = utils.build_critic(args, data, env, summary_writer)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            # policy.set_param_values(params['policy'])
            critic.network.set_param_values(params['critic'])

            normalized_env = hgail.misc.utils.extract_normalizing_env(env)

            if normalized_env is not None:
                normalized_env._obs_mean = params['normalzing']['obs_mean']
                normalized_env._obs_var = params['normalzing']['obs_var']

            rate = 0
            for i in range(data['actions'].shape[0]):
                ob = np.expand_dims(data['observations'][i, :, :], axis=0)

                # (2150, 1010, 2)
                ac = np.expand_dims(data['actions'][i, :, :], axis=0)
                paths = [{'observations': ob[:, i, :], 'actions': ac[:, i, :]} for i in range(ac.shape[1])]
                obs = np.concatenate([d['observations'] for d in paths], axis=0)
                acts = np.concatenate([d['actions'] for d in paths], axis=0)

            
                # normalize
                if critic.dataset.observation_normalizer:
                    obs = critic.dataset.observation_normalizer(obs)
                if critic.dataset.action_normalizer:
                    acts = critic.dataset.action_normalizer(acts)

                # compute rewards
                rewards = critic.network.forward(obs, acts, deterministic=True)
                
                # output as a list of numpy arrays, each of len equal to the rewards of 
                # the corresponding trajectory
                # path_lengths = [len(d['rewards']) for d in paths]
                # critic_rewards = hgail.misc.utils.batch_to_path_rewards(rewards, path_lengths)
                print(min(rewards), max(rewards))

                rate += abs(min(rewards) - 16) < 1 and abs(max(rewards) - 27) < 1

            print(rate / data['actions'].shape[0])

