import h5py, os, pdb
import numpy as np
import utils
import tensorflow as tf
import hyperparams
from julia_env.julia_env import JuliaEnv

NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1
}

def load_validate_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf):
    
    # loading varies based on dataset type
    x, feature_names = utils.load_x_feature_names(filepath, ngsim_filename)

    # no need to flatten 
    obs = x
    act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:, :, act_idxs]

    if normalize_data:
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(np.mean(x, axis=0),axis=0, keepdims=True)
    x = x - mean
    x_flatten = np.reshape(x, [-1, 66])
    std = np.std(x_flatten, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std

def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x


def get_ground_truth():
	# filepath = '../../data/trajectories/ngsim_22agents.h5'
	# ngsim_filename='trajdata_i101-22agents-0750am-0805am.txt'
    # filepath = '../../data/trajectories/ngsim.h5'
    # ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt'
    # x, feature_names = load_x_feature_names(filepath, ngsim_filename)
    # clip_std_multiple = 10.
    # x, obs_mean, obs_std = normalize(x, clip_std_multiple)
    # # x.shape = [2150, 1010, 66]
    # hyperparamters
    '''Namespace(batch_size=10000, critic_batch_size=1000, critic_dropout_keep_prob=0.8, critic_grad_rescale=40.0, critic_hidden_layer_dims=(128, 128, 64), critic_learning_rate=0.0004, decay_reward=False, discount=0.95, do_curriculum=False, env_H=200, env_action_repeat=1, env_multiagent=False, env_primesteps=50, env_reward=0, exp_dir='../../data/experiments', exp_name='singleagent_def_3', expert_filepath='../../data/trajectories/ngsim.h5', gradient_penalty=2.0, itrs_per_decay=25, latent_dim=4, load_params_init='NONE', max_path_length=1000, n_critic_train_epochs=40, n_envs=1, n_envs_end=50, n_envs_start=10, n_envs_step=10, n_itr=1000, n_recognition_train_epochs=30, ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt', normalize_clip_std_multiple=10.0, params_filepath='', policy_mean_hidden_layer_dims=(128, 128, 64), policy_recurrent=True, policy_std_hidden_layer_dims=(128, 64), recognition_hidden_layer_dims=(128, 64), recognition_learning_rate=0.0005, recurrent_hidden_dim=64, remove_ngsim_veh=False, render_every=25, reward_handler_critic_final_scale=1.0, reward_handler_max_epochs=100, reward_handler_recognition_final_scale=0.2, reward_handler_use_env_rewards=True, scheduler_k=20, trpo_step_size=0.01, use_critic_replay_memory=True, use_infogail=False, validator_render=False, vectorize=True)'''

    # build components
    # env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=True)
    act_low = np.array([-4, -0.15])
    act_high= np.array([4,  0.15])
    data = load_validate_data(
        '../../data/trajectories/ngsim.h5', 
        act_low=act_low, 
        act_high=act_high, 
        min_length= 200 + 50,
        clip_std_multiple=10.0,
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt'
    )
    return data


if __name__ == '__main__':
    get_ground_truth()

def build_ngsim_env(
        args,
        exp_dir='/tmp', 
        alpha=0.001,
        vectorize=False,
        render_params=None,
        videoMaking=False):
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filepaths = [os.path.join(basedir, 'trajdata_i101_trajectories-0750am-0805am.txt')]
    if render_params is None:
        render_params = dict(
            viz_dir=os.path.join(exp_dir, 'imitate/viz'),
            zoom=5.
        )
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=200,
        primesteps=50,
        action_repeat=1,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=render_params,
        n_envs=1,
        n_veh=1,
        remove_ngsim_veh=False,
        reward=0
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must 
    # also be true
    env_id = 'MultiagentNGSIMEnv'

    env = JuliaEnv(
        env_id=env_id,
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = None
    return env, low, high

# 'relative_offset', 'relative_heading', 'velocity', 'length',
# 'width', 'lane_curvature', 'markerdist_left', 'markerdist_right',
# 'accel' 8 , 'jerk', 'turn_rate_global' 10, 'angular_rate_global'11,
# 'turn_rate_frenet'12, 'angular_rate_frenet'13, 'timegap',
# 'timegap_is_avail', 'time_to_collision',
# 'time_to_collision_is_avail', 'is_colliding', 'out_of_lane',
# 'negative_velocity', 'distance_road_edge_left',
# 'distance_road_edge_right', 'lidar_1', 'lidar_2', 'lidar_3',
# 'lidar_4', 'lidar_5', 'lidar_6', 'lidar_7', 'lidar_8', 'lidar_9',
# 'lidar_10', 'lidar_11', 'lidar_12', 'lidar_13', 'lidar_14',
# 'lidar_15', 'lidar_16', 'lidar_17', 'lidar_18', 'lidar_19',
# 'lidar_20', 'rangerate_lidar_1', 'rangerate_lidar_2',
# 'rangerate_lidar_3', 'rangerate_lidar_4', 'rangerate_lidar_5',
# 'rangerate_lidar_6', 'rangerate_lidar_7', 'rangerate_lidar_8',
# 'rangerate_lidar_9', 'rangerate_lidar_10', 'rangerate_lidar_11',
# 'rangerate_lidar_12', 'rangerate_lidar_13', 'rangerate_lidar_14',
# 'rangerate_lidar_15', 'rangerate_lidar_16', 'rangerate_lidar_17',
# 'rangerate_lidar_18', 'rangerate_lidar_19', 'rangerate_lidar_20',
# 'fore_fore_dist', 'fore_fore_relative_vel', 'fore_fore_accel'