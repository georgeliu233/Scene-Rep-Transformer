from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB,Waypoints
from smarts.core.controllers import ActionSpaceType
import gym
import numpy as np

AGENT_ID = 'Agent-007'

ep_step = 0
def reward_adapter(env_obs, env_reward):
    global ep_step
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if (env_obs.events.collisions) else 0
    step = -1 if ep_step>=0.5*800 else 0
    #env_obs.events.reached_max_episode_steps
    ep_step+=1

    if  (env_obs.events.collisions 
        or env_obs.events.reached_goal 
        or env_obs.events.agents_alive_done 
        or env_obs.events.reached_max_episode_steps
        or env_obs.events.off_road):
        ep_step = 0
    # return env_reward
    return goal + crash #-1/800#+step/(800*0.5)#+0.001*step #+0.001*env_reward #+ 0.0005*progress

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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", help="algorithm to run",type=str)
args = parser.parse_args()
f = '/home/haochen/SMARTS/'

if args.scenario == 'left_turn':
    scenario_path = [f+'scenarios/left_turn_new']
    max_episode_steps = 400
elif args.scenario == 'r':
    scenario_path = [f+'scenarios/roundabout']
    max_episode_steps = 1000
elif args.scenario == 'cross':
    scenario_path = [f+'scenarios/double_merge/cross']
    max_episode_steps = 600
elif args.scenario == 're':
    scenario_path = [f+'scenarios/roundabout_easy']
    max_episode_steps = 400
elif args.scenario == 'rm':
    scenario_path = [f+'scenarios/roundabout_medium']
    max_episode_steps = 600
else:
    raise NotImplementedError

agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    neighborhood_vehicles=True,
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

save_dir = '/home/haochen/TPDM_transformer/DT/data/'+'data_'+scenario_path[0].split('/')[-1] +'.pkl'
# define agent specs
agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=0)
env.agent_id = AGENT_ID
env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

from trainer import Collector

collector = Collector(env, save_dir, collect_trajs=500, max_traj=max_episode_steps, num_neighbors=5)

collector.collect()
