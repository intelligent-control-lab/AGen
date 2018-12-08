
import numpy as np
import os
import tensorflow as tf

from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp, ConjugateGradientOptimizer

from hgail.algos.gail import GAIL

import auto_validator
import hyperparams
import utils

# increase the number of agents, running n_itrs each time, starting at start_n_envs and ending at end...
def do_curriculum(args):
    # key insight is to track where it was saved to previously, and load from there.
    # have that returned by decaying rewards, for non-decaying, manually track.
    base_exp_name = args.exp_name
    n_itr_each = args.n_itr
    
    # initial params filepath to allow continuation mid curriculum training
    if args.load_params_init != 'NONE':
        print("starting at:", args.load_params_init)
        load_exp_name = base_exp_name + "_" + args.load_params_init
        itr_npz = n_itr_each
        if args.decay_reward:
            load_exp_name += "_" + str(n_itr_each - args.itrs_per_decay)
            itr_npz = args.itrs_per_decay
        args.params_filepath = os.path.join(
            args.exp_dir,
            load_exp_name,
            'imitate/log/itr_{}.npz'.format(itr_npz)
        )
        print("Full path:", args.params_filepath)

    for n_envs in range(args.n_envs_start, args.n_envs_end + args.n_envs_step, args.n_envs_step):
        args.n_envs = n_envs
        args.exp_name = base_exp_name + '_' + str(n_envs)
        
        n_itrs = args.n_itr
        if args.decay_reward:
            do_decaying_reward(args)
            args.exp_name += '_' + str(n_itr_each - args.itrs_per_decay)
            n_itrs = args.itrs_per_decay
        else:
            run(args)
	
        # update params filepath
        args.params_filepath = os.path.join(
            args.exp_dir,
            args.exp_name, # previous experiment name
            'imitate/log/itr_{}.npz'.format(n_itrs)
        )



# to handle the need to decay rewards (increasing rewards is also simple)
# important to reset the args after done decaying.
def do_decaying_reward(args):
    # store these to reset to later. 
    initial_n_itr = args.n_itr
    initial_reward = args.env_reward
    experiment_name = args.exp_name 

    itrs = list(range(0, args.n_itr, args.itrs_per_decay))
    args.n_itr = args.itrs_per_decay
    R_values = [int(r) for r in np.linspace(args.env_reward, 0, len(itrs))] 
    # int() cast to make sure no inexact errors
    
    for i in range(len(itrs)):
        args.env_reward = R_values[i]
        print("Running with R:", args.env_reward)
        args.exp_name = experiment_name + '_' + str(itrs[i])
        run(args)
        args.params_filepath = os.path.join(
            args.exp_dir,
            args.exp_name, # previous experiment name
            'imitate/log/itr_{}.npz'.format(args.itrs_per_decay)
        )
        tf.reset_default_graph()

    # reset number of iterations and env reward
    args.n_itr = initial_n_itr
    args.env_reward = initial_reward
    args.exp_name = experiment_name
        
# the normal function
def run(args):
    '''Namespace(batch_size=10000, critic_batch_size=1000, critic_dropout_keep_prob=0.8, critic_grad_rescale=40.0, critic_hidden_layer_dims=(128, 128, 64), critic_learning_rate=0.0004, decay_reward=False, discount=0.95, do_curriculum=False, env_H=200, env_action_repeat=1, env_multiagent=False, env_primesteps=50, env_reward=0, exp_dir='../../data/experiments', exp_name='singleagent_def_3', expert_filepath='../../data/trajectories/ngsim.h5', gradient_penalty=2.0, itrs_per_decay=25, latent_dim=4, load_params_init='NONE', max_path_length=1000, n_critic_train_epochs=40, n_envs=1, n_envs_end=50, n_envs_start=10, n_envs_step=10, n_itr=1000, n_recognition_train_epochs=30, ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt', normalize_clip_std_multiple=10.0, params_filepath='', policy_mean_hidden_layer_dims=(128, 128, 64), policy_recurrent=True, policy_std_hidden_layer_dims=(128, 64), recognition_hidden_layer_dims=(128, 64), recognition_learning_rate=0.0005, recurrent_hidden_dim=64, remove_ngsim_veh=False, render_every=25, reward_handler_critic_final_scale=1.0, reward_handler_max_epochs=100, reward_handler_recognition_final_scale=0.2, reward_handler_use_env_rewards=True, scheduler_k=20, trpo_step_size=0.01, use_critic_replay_memory=True, use_infogail=False, validator_render=False, vectorize=True)'''
    
    print("loading from:", args.params_filepath)
    print("saving to:", args.exp_name)
    exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
    saver_dir = os.path.join(exp_dir, 'imitate', 'log')
    saver_filepath = os.path.join(saver_dir, 'checkpoint')
    np.savez(os.path.join(saver_dir, 'args'), args=args)
    summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))

    # build components
    env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=args.vectorize)
    print(act_low, act_high)
    data = utils.load_data(
        args.expert_filepath, 
        act_low=act_low, 
        act_high=act_high, 
        min_length=args.env_H + args.env_primesteps,
        clip_std_multiple=args.normalize_clip_std_multiple,
        ngsim_filename=args.ngsim_filename
    )
    critic = utils.build_critic(args, data, env, summary_writer)
    policy = utils.build_policy(args, env)
    recognition_model = utils.build_recognition_model(args, env, summary_writer)
    baseline = utils.build_baseline(args, env)
    reward_handler = utils.build_reward_handler(args, summary_writer)
    validator = auto_validator.AutoValidator(
        summary_writer, 
        data['obs_mean'], 
        data['obs_std'],
        render=args.validator_render,
        render_every=args.render_every,
        flat_recurrent=args.policy_recurrent
    )

    # build algo 
    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
    sampler_args = dict(n_envs=args.n_envs) if args.vectorize else None
    if args.policy_recurrent:
        optimizer = ConjugateGradientOptimizer(
            max_backtracks=50,
            hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)
        )
    else:
        optimizer = None
    algo = GAIL(
        critic=critic,
        recognition=recognition_model,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        validator=validator,
        batch_size=args.batch_size,
        max_path_length=args.max_path_length,
        n_itr=args.n_itr,
        discount=args.discount,
        step_size=args.trpo_step_size,
        saver=saver,
        saver_filepath=saver_filepath,
        force_batch_sampler=False if args.vectorize else True,
        sampler_args=sampler_args,
        snapshot_env=False,
        plot=False,
        optimizer=optimizer,
        optimizer_args=dict(
            max_backtracks=50,
            debug_nan=True
        )
    )

    # run it
    with tf.Session() as session:
        
        # running the initialization here to allow for later loading
        # NOTE: rllab batchpolopt runs this before training as well 
        # this means that any loading subsequent to this is nullified 
        # you have to comment of that initialization for any loading to work
        session.run(tf.global_variables_initializer())

        # loading
        if args.params_filepath != '':
            algo.load(args.params_filepath)
        print("I have been here.\n")                      
        # run training
        algo.train(sess=session)


# setup
args = hyperparams.parse_args()
if args.do_curriculum:
    do_curriculum(args)
elif args.decay_reward:
    do_decaying_reward(args)
else:
    print("in run seesion.")
    run(args)

