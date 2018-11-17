import pdb
import h5py
import numpy as np

NGSIM_FILENAME_TO_ID = {
    'trajdata_i101-22agents-0750am-0805am.txt': 1
}

def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std

def load_x_feature_names(filepath, ngsim_filename):
    f = h5py.File(filepath, 'r')
    xs = []
    traj_id = NGSIM_FILENAME_TO_ID[ngsim_filename]
    # in case this nees to allow for multiple files in the future
    traj_ids = [traj_id]
    for i in traj_ids:
        if str(i) in f.keys():
            xs.append(f[str(i)])
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    pdb.set_trace()
    x = np.concatenate(xs)
    feature_names = f.attrs['feature_names']
    return x, feature_names

if __name__ == '__main__':
	filepath = '../../data/trajectories/ngsim_22agents.h5'
	ngsim_filename='trajdata_i101-22agents-0750am-0805am.txt'
	x, feature_names = load_x_feature_names(filepath, ngsim_filename)
	
	clip_std_multiple = 10.
	x, obs_mean, obs_std = normalize(x, clip_std_multiple)
    # x.shape = [2150, 1010, 66]
	pdb.set_trace()
	


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