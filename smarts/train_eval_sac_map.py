from sac import SAC

from tf2rl.experiments.trainer import Trainer
from trainer import Trainer

import tensorflow as tf
import gym
import numpy as np
import glob
import random
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB,Waypoints
from smarts.core.controllers import ActionSpaceType
import os
from sac_actor_critic_policy import GaussianActorCritic
from collections import deque
from obs_adapter import NeighbourObsAdapter

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.set_visible_devices(devices=gpus[-2], device_type='GPU')
# tf.config.experimental.set_memory_growth(gpus[-1], True)


#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

AGENT_ID = 'Agent-007'


LSTM=False
CNN_INPUTS=True
EGO_SURR=False
BPTT=False

USE_MASK=True
multi_selection=False
multi_actions=False
ensemble = False
representations = False

N_steps=10
future_steps=0
pred_trajs=1
multi_pred=0

dim=5
neighbors = 5
planned_path = 10

obs_adapter = NeighbourObsAdapter('map',N_steps,dim,neighbors,return_map=True,prediction=True,planned_path=planned_path)
test_obs_adapter = NeighbourObsAdapter('map',N_steps,dim,neighbors,return_map=True,prediction=True,planned_path=planned_path)

OBSERVATION_SPACE = obs_adapter.OBSERVATION_SPACE

print('mode:{},obs_space:{}'.format('map',OBSERVATION_SPACE))

# reward function
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

    return goal + crash 

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


#### RL training ####
parser = Trainer.get_argument()

args = parser.parse_args()
args.scenario = 'left_turn'
args.max_steps = 10e4
args.save_summary_interval = 1000
args.save_model_interval = 10000

args.show_progress = False
args.use_prioritized_rb = False
args.n_experiments = 3
args.n_steps=3
args.use_nstep_rb=True
args.test_episodes=50
args.model_dir=''
args.gpu=0


# define scenario
f = ''

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

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=Waypoints(planned_path),
    neighborhood_vehicles=NeighborhoodVehicles(radius=None),
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

# define agent specs
agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

test_agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=observation_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

neighbor_interface = AgentInterface(
            max_episode_steps=None,
            action=ActionSpaceType.Lane,
            waypoints=Waypoints(planned_path))
neighbor_spec = neighbor_interface
test_neighbor_spec = neighbor_interface

params={
    'bptt':BPTT,
    'ego_surr':False,
    'use_trans':False,
    'cnn':False,
    'neighbours':neighbors,
    'time_step':N_steps,
    'make_rotation':True,
    'make_prediction':bool(future_steps>0),
    'make_prediction_q':False,
    'make_prediction_value':False,
    'traj_nums':pred_trajs,
    'traj_length':future_steps,
    'use_map':False,
    'state_input':False,
    'LSTM':False,
    'cnn_lstm':False,
    'path_length':planned_path,
    'head_num':1,
    'n_steps':args.n_steps,
    'N_steps':N_steps,
    'future_step':future_steps,
    'comments':'pure actor critic',
    'use_hier':True,
    'random_aug':False,
    'no_ego_fut':False,
    'no_neighbor_fut':False
}

if ensemble:
   params['ensembles']=True

print(params)

discount = 0.99
p_dis = discount ** (1/args.n_steps)

# create env
env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=1)
env.observation_space = OBSERVATION_SPACE
env.action_space = ACTION_SPACE
env.agent_id = AGENT_ID

test_env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: test_agent_spec}, headless=True, seed=1)
test_env.observation_space = OBSERVATION_SPACE
test_env.action_space = ACTION_SPACE
test_env.agent_id = AGENT_ID

policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                batch_size=32, auto_alpha=True, memory_capacity=int(2e4),n_warmup=500,lr=1e-4,
                gpu=0,make_prediction=future_steps>0,pred_trajs=pred_trajs,discount=discount,params=params,multi_selection=multi_selection
                ,multi_actions=multi_actions,actor_critic=True,representations=representations,ensembles=ensemble,
                update_interval=10000)
bptt_hidden = 0 if not BPTT else 256
trainer = Trainer(policy=policy,env=env,args=args,test_env=env,save_path=f'ANeighborCLsac_map_glb5_{args.scenario}',
                        use_mask=USE_MASK,bptt_hidden=bptt_hidden,use_ego=False,
                        make_predictions=future_steps,use_map=True,obs_adapter=obs_adapter.obs_adapter,
                        test_obs_adapter=test_obs_adapter.obs_adapter,neighbor_spec=neighbor_spec,test_neighbor_spec=test_neighbor_spec,
                        path_length=planned_path,params=params,neighbors=neighbors,multi_prediction=multi_pred,
                        multi_selection=multi_selection,pred_future_state=representations,skip_timestep=3,
                        test_pca=False)

# begin training
# trainer()
trainer.evaluate_policy_continuously(plot_map_mode=False,map_dir='./'+scenario_path[0]+'/')


# close env
env.close()