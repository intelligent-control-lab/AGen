
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

plt.style.use("ggplot")

def online_adaption(
        env, 
        policy, 
        max_steps,  
        obs,
        mean,
        render=False, 
        env_kwargs=dict(),
        lbd=0.99,
        adapt_steps=1):

    if len(obs.shape) == 2:
        obs = np.expand_dims(obs, axis=0)
        mean = np.expand_dims(mean, axis=0)

    theta = np.load('theta.npy')
    theta = np.mean(theta)

    x = env.reset(**env_kwargs)
    
    n_agents = x.shape[0]
    dones = [True] * n_agents
    predicted_trajs, adapnets = [], []
    policy.reset(dones)
    prev_actions, prev_hiddens = None, None
    
    max_steps = min(1000, obs.shape[1])
        
    mean = np.expand_dims(mean, axis=2)
    prev_hiddens = np.zeros([n_agents,64])

    param_length = 65 if adapt_steps == 1 else 195

    for i in range(n_agents):
        adapnets.append(rls.rls(lbd, theta, param_length, 2))

    avg = 0
    for step in range(max_steps-1):
        if step % 100 == 0:
            print(step)

        start = time.time()
        a, a_info, hidden_vec = policy.get_actions_with_prev(obs[:,step,:], mean[:, step,:], prev_hiddens)

        if adapt_steps == 1:
            adap_vec = hidden_vec
            print("one step")
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, obs[:,step,:]), axis=1)

        adap_vec = np.expand_dims(adap_vec, axis=1)

        for i in range(n_agents):
            adapnets[i].update(adap_vec[i], mean[i,step+1,:])
            adapnets[i].draw.append(adapnets[i].theta[6,1])

        prev_actions, prev_hiddens = a, hidden_vec
        
        traj = prediction(env_kwargs, obs[:,step+1,:], adapnets, env, policy, prev_hiddens, n_agents, adapt_steps)

        predicted_trajs.append(traj)
        d = np.stack([adapnets[i].draw for i in range(n_agents)])
        end = time.time()
        avg += (start - end)

    print(avg / (max_steps-1))
    for i in range(n_agents):
        plt.plot(range(step+1), d[i,:])
    plt.show()
    
    return predicted_trajs

def prediction(env_kwargs, x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps):
    traj = hgail.misc.simulation.Trajectory()
    predict_span = 200
    for i in range(predict_span):
        a, a_info, hidden_vec = policy.get_actions(x)

        if adapt_steps == 1:
            adap_vec = hidden_vec
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, x), axis=1)

        means = np.zeros([n_agents, 2])
        log_std = np.zeros([n_agents, 2])
        for i in range(x.shape[0]):
            means[i] = adapnets[i].predict(np.expand_dims(adap_vec[i], 0))
            log_std[i] = np.log(np.std(adapnets[i].theta, axis=0))

        prev_hiddens = hidden_vec

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_std) + means

        nx, r, dones, e_info = env.step(actions)
        traj.add(x, actions, r, a_info, e_info)
        if any(dones): break
        x = nx
        
    y = env.reset(**env_kwargs)

    return traj.flatten()

def collect_trajectories(
        args,  
        params, 
        egoids, 
        starts,
        trajlist,
        pid,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed,
        lbd,
        adapt_steps):
    
    env, _, _ = env_fn(args, alpha=0.)
    policy = policy_fn(args, env)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(params[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy.set_param_values(params['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # collect trajectories
        nids = len(egoids)

        if args.env_multiagent:

            data = validate_utils.get_multiagent_ground_truth()
        else:
            data = validate_utils.get_ground_truth()
            sample = np.random.choice(data['observations'].shape[0], 2)

        kwargs = dict()
        if args.env_multiagent:
            # I add not because single simulation has no orig_x etc.
            
            if random_seed:
                kwargs = dict(random_seed=random_seed+egoid)

            traj = online_adaption(
                env, 
                policy, 
                max_steps=max_steps,
                obs=data['observations'],
                mean=data['actions'],
                env_kwargs=kwargs,
                lbd=lbd,
                adapt_steps=adapt_steps
            )
            trajlist.append(traj)
        else:
            for i in sample:
            
                sys.stdout.write('\rpid: {} traj: {} / {}'.format(pid, i, nids))

                traj = online_adaption(
                    env, 
                    policy, 
                    max_steps=max_steps,
                    obs=data['observations'][i, :,:],
                    mean=data['actions'][i,:,:],
                    env_kwargs=kwargs,
                    lbd=lbd,
                    adapt_steps=adapt_steps
                )
                trajlist.append(traj)
                

    return trajlist

def parallel_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None,
        lbd = 0.99,
        adapt_steps = 1):
    # build manager and dictionary mapping ego ids to list of trajectories
    manager = mp.Manager()
    trajlist = manager.list()

    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else validate_utils.build_policy
    
    # partition egoids 
    proc_egoids = utils.partition_list(egoids, n_proc)

    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)

    # run collection
    results = []
    for pid in range(n_proc):
        res = pool.apply_async(
            collect_trajectories,
            args=(
                args, 
                params, 
                proc_egoids[pid], 
                starts,
                trajlist, 
                pid,
                env_fn,
                policy_fn,
                max_steps,
                use_hgail,
                random_seed,
                lbd,
                adapt_steps
            )
        )
        results.append(res)

    # wait for the processes to finish
    [res.get() for res in results]
    pool.close()
    # let the julia processes finish up
    time.sleep(10)
    return trajlist

def single_process_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc, 
        env_fn=utils.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    '''
    This function for debugging purposes
    '''
    # build list to be appended to 
    trajlist = []
    
    # set policy function
    policy_fn = utils.build_hierarchy if use_hgail else validate_utils.build_policy
    tf.reset_default_graph()

    # collect trajectories in a single process
    collect_trajectories(
        args, 
        params, 
        egoids, 
        starts,
        trajlist, 
        n_proc,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed
    )
    return trajlist    

def collect(
        egoids,
        starts,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        max_steps=200,
        collect_fn=parallel_collect_trajectories,
        random_seed=None,
        lbd = 0.99,
        adapt_steps = 1):
    '''
    Description:
        - prepare for running collection in parallel
        - multiagent note: egoids and starts are not currently used when running 
            this with args.env_multiagent == True 
    '''
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/log/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)
    # validation setup 
    validation_dir = os.path.join(exp_dir, 'imitate', 'validation')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_APSGAIL.npz'.format(
        args.ngsim_filename.split('.')[0]))

    with Timer():
        trajs = collect_fn(
            args, 
            params, 
            egoids, 
            starts,
            n_proc,
            max_steps=max_steps,
            use_hgail=use_hgail,
            random_seed=random_seed,
            lbd = 0.99,
            adapt_steps = 1
        )

    utils.write_trajectories(output_filepath, trajs)

def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=utils.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data/')
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    ids_filepath = os.path.join(basedir, ids_filename)
    if not os.path.exists(ids_filepath):
        # this should create the ids file
        env_fn(args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)

    # we want to sample start times uniformly from the range of possible values 
    # but we also want these start times to be identical for every model we 
    # validate. So we sample the start times a single time, and save them.
    # if they exist, we load them in and reuse them
    start_times_filename = filename.replace('.txt', '-index-{}-starts.h5'.format(offset))
    start_times_filepath = os.path.join(basedir, start_times_filename)
    # check if start time filepath exists
    if os.path.exists(start_times_filepath):
        # load them in
        starts = np.array(h5py.File(start_times_filepath, 'r')['starts'].value)
    # otherwise, sample the start times and save them
    else:
        ids_file = h5py.File(ids_filepath, 'r')
        ts = ids_file['ts'].value
        # subtract offset gives valid end points
        te = ids_file['te'].value - offset
        starts = np.array([np.random.randint(s,e+1) for (s,e) in zip(ts,te)])
        # write to file
        starts_file = h5py.File(start_times_filepath, 'w')
        starts_file.create_dataset('starts', data=starts)
        starts_file.close()

    # create a dict from id to start time
    id2starts = dict()
    for (egoid, start) in zip(ids, starts):
        id2starts[egoid] = start

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, id2starts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='../../data/experiments/gail/')
    parser.add_argument('--params_filename', type=str, default='itr_2000.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=10)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)
    parser.add_argument('--lbd', type=float, default=0.99)
    parser.add_argument('--adapt_steps', type=int, default=1)

    run_args = parser.parse_args()

    args_filepath = os.path.join(run_args.exp_dir, 'imitate/log/args.npz')
    args = hyperparams.load_args(args_filepath)
    
    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    if run_args.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories

    filenames = [
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]

    if run_args.n_envs:
        args.n_envs = run_args.n_envs
    # args.env_H should be 200
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
        collect(
            egoids,
            starts,
            args,
            exp_dir=run_args.exp_dir,
            max_steps=200,
            params_filename=run_args.params_filename,
            use_hgail=run_args.use_hgail,
            n_proc=run_args.n_proc,
            collect_fn=collect_fn,
            random_seed=run_args.random_seed,
            lbd = run_args.lbd,
            adapt_steps = run_args.adapt_steps
        )