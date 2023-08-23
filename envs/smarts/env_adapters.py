import numpy as np
from smarts.core.utils.math import position_to_ego_frame

def reward_adapter(env_obs, env_reward):
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if (env_obs.events.collisions) else 0
    return goal + crash 

def reward_adapter_mf(env_obs, env_reward):

    ego_speed = env_obs.ego_vehicle_state.speed
    speed_reward = min(ego_speed, 10 - ego_speed)
    goal = 10 if env_obs.events.reached_goal else 0
    crash = -10 if env_obs.events.collisions else 0
    step = - 0.1 
    steering = - 0.5 * (float(env_obs.ego_vehicle_state.heading))**2
  
    return goal + crash + speed_reward + step + steering

def reward_adapter_hier(env_obs, env_reward):
    
    goal = 100 if env_obs.events.reached_goal else 0
    crash = -50 if (env_obs.events.collisions or env_obs.events.reached_max_episode_steps) else 0
    off_road = -1 if env_obs.events.off_road else 0

    neighbors = env_obs.neighborhood_vehicle_states
    ego = env_obs.ego_vehicle_state
    position_differences = np.array([(ego.position[0]-neighbor.ego_vehicle_state.position[0])**2 +
                    (ego.position[1]-neighbor.ego_vehicle_state.position[1])**2 for neighbor in neighbors])
    nearest_indexes = np.argmin(position_differences)
    nearest_vehicle_dist = np.sqrt(position_differences[nearest_indexes])

    if nearest_vehicle_dist > 20:
        d_lon, d_lat = 20, 3.5
    else:
        nearest_vehicle = neighbors[nearest_indexes]
        ego_steering = ego.heading + np.pi/2
        trans_pos = position_to_ego_frame(np.array(nearest_vehicle.position), np.array(ego.position), ego_steering)
        d_lon, d_lat = trans_pos[0], trans_pos[1]

    safety_dist = 0.02*d_lon + 0.01*d_lat

    # env_reward for SMARTS is the dist travelled
    dist_travelled_reward = - 1 / (env_reward + 1)
  
    return goal + crash + off_road + safety_dist + dist_travelled_reward

# action space
def action_adapter(model_action): 
    speed = model_action[0] # output (-1, 1)
    speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10)
    
    speed = np.clip(speed, 0, 10)
    model_action[1] = np.clip(model_action[1], -1, 1)

    # discretization
    if model_action[1] < -1/3:
        lane = -1
    elif model_action[1] > 1/3:
        lane = 1
    else:
        lane = 0

    return (speed, lane)

# information
def info_adapter(observation, reward, info):
    return observation.events

def observation_adapter(observation):
    return observation