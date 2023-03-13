
import numpy as np
from smarts.core.utils.math import position_to_ego_frame, wrap_value
import pickle
import math

class Collector:
    def __init__(self, env, save_dir, collect_trajs=1000, max_traj=200, num_neighbors=5):
        self.env = env
        self.save_dir = save_dir
        self.max_traj = max_traj
        self.num_neighbors = num_neighbors
        self.collect_trajs = collect_trajs
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.buffer = {
            'states':[],# np.zeros((self.max_traj, 1+self.num_neighbors, 10), np.float32),
            'actions': [],#np.zeros((self.max_traj, 2), np.float32),
            'rewards': [],
            'dones': []
        }
        self.timestep = 0
        return self.observation(observation)

    def ego_process(self, ego):
        self.ego_id = ego.id
        ego_position = self.transform(ego.position)[:2]
        ego_heading = self.adjust_heading(ego.heading+np.pi/2)
        ego_linear_velocity = self.ego_frame_dynamics(ego.linear_velocity)
        ego_linear_acceleration = self.ego_frame_dynamics(ego.linear_acceleration)
        ego_target = self.transform(ego.mission.goal.position)[:2]
        ego_to_target = np.linalg.norm(ego_target)
        ego_speed = ego.speed
        ego_angular = ego.yaw_rate if ego.yaw_rate else 0
        
        # [x, y, v_x, v_y, a_x, a_y, heading, yaw rate, speed, to_target]
        ego_state = np.concatenate([ego_position,  [ego_heading, ego_speed, ego_to_target]])

        return ego_state

    def neighbor_process(self, env_obs):
        neighbors_state = np.zeros(shape=(self.num_neighbors, 5))

        neighbors = {}
        i = 0
        for neighbor in env_obs.neighborhood_vehicle_states:
            neighbors[i] = neighbor.position[:2]
            i += 1
        
        sorted_neighbors = sorted(neighbors.items(), key=lambda item: np.linalg.norm(item[1] - self.current_pos[0][:2]))
        sorted_neighbors = sorted_neighbors[:self.num_neighbors]
        
        neighbor_ids = [neighbor[0] for neighbor in sorted_neighbors]

        i = 0
        for neigbhbor_id in neighbor_ids:
            neighbor = env_obs.neighborhood_vehicle_states[neigbhbor_id]
            neighbor_position = self.transform(neighbor.position)[:2]
            neighbor_heading = self.adjust_heading(neighbor.heading+np.pi/2)
            neighbor_speed = neighbor.speed
            neighbor_to_ego = np.linalg.norm(neighbor_position)

            # [x, y, v_x, v_y, a_x, a_y, heading, yaw rate, speed]
            neighbor_state = np.concatenate([neighbor_position, [neighbor_heading, neighbor_speed, neighbor_to_ego]])
            neighbors_state[i] = neighbor_state
            i += 1

        return neighbors_state
    
    def observation(self,obs):
        env_obs = obs[self.env.agent_id]
        self.current_pos = (env_obs.ego_vehicle_state.position, env_obs.ego_vehicle_state.heading+np.pi/2)
        ego_state = self.ego_process(env_obs.ego_vehicle_state)
        neighbor_state = self.neighbor_process(env_obs)
        return np.concatenate([ego_state[np.newaxis,...], neighbor_state], axis=0)
    
    def observation_adapter(self,env_obs):
        self.current_pos = (env_obs.ego_vehicle_state.position, env_obs.ego_vehicle_state.heading+np.pi/2)
        ego_state = self.ego_process(env_obs.ego_vehicle_state)
        neighbor_state = self.neighbor_process(env_obs)
        return np.reshape(np.concatenate([ego_state[np.newaxis,...], neighbor_state], axis=0), (-1))
    
    def step(self):
        action = self.env.action_space.sample()
        next_obs, reward, done, info = self.env.step({self.env.agent_id: action})

        next_obs = self.observation(next_obs)
        reward = reward[self.env.agent_id]
        done = done[self.env.agent_id]
        info = info[self.env.agent_id]

        return next_obs, reward, done, info, action
    
    def episode(self):
        t_step = 0
        obs = self.reset()
        self.buffer['states'].append(obs)
        for t_step in range(self.max_traj):
            next_obs, reward, done, info, action = self.step()
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['dones'].append(done)
            if done:
                break
            self.buffer['states'].append(obs)
        return info, t_step
    
    def collect(self):
        ep = 0
        collect_data = []
        for t in range(self.collect_trajs):
            info, t_step = self.episode()
            success = 1 if info.reached_goal else 0
            print (f' Episode: {t} Success: {success} Timestep: {t_step}')
            if success:
                collect_data.append(self.buffer)
        print(len(collect_data),'saving...')

        with open(self.save_dir,'wb') as writer:
            pickle.dump(collect_data, writer)
        print('Saved!',self.save_dir)

    def transform(self, v):
        return position_to_ego_frame(v, self.current_pos[0], self.current_pos[1])

    def adjust_heading(self, h):
        return wrap_value(h - self.current_pos[1], -math.pi, math.pi)

    def ego_frame_dynamics(self, v):
        ego_v = v.copy()
        ego_v[0] = v[0] * np.cos(self.current_pos[1]) + v[1] * np.sin(self.current_pos[1])
        ego_v[1] = v[1] * np.cos(self.current_pos[1]) - v[0] * np.sin(self.current_pos[1])

        return ego_v

