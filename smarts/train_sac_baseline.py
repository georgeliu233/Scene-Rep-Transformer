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
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.core.controllers import ActionSpaceType
import os
from sac_actor_critic_policy import GaussianActorCritic
from collections import deque
from utils import NeighbourAgentBuffer

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.set_visible_devices(devices=gpus[-2], device_type='GPU')



#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

AGENT_ID = 'Agent-007'


LSTM=True
CNN_INPUTS=True
N_steps=3
future_steps=0
pred_trajs=5

dim=5
neighbors = 5


if CNN_INPUTS:
    if not LSTM:
        OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 3*N_steps))
    else:
        OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(N_steps,80, 80, 3))
else:
    OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(neighbors+1,N_steps,dim))
    
    neighbours_buffer = NeighbourAgentBuffer(state_shape=OBSERVATION_SPACE.shape,hist_length=N_steps,future_length=10,query_mode='history_only')
    test_neighbours_buffer = NeighbourAgentBuffer(state_shape=OBSERVATION_SPACE.shape,hist_length=N_steps,future_length=10,query_mode='history_only')

states = deque(maxlen=N_steps)
epi_steps = 0
test_states = deque(maxlen=N_steps)
test_epi_steps = 0

def observation_adapter(env_obs):
    global states
    global epi_steps

    new_obs = env_obs.top_down_rgb[1] / 255.0
    
    states.append(new_obs)
    res = np.array(list(states))

    if res.shape[0]<N_steps:
        res = np.concatenate((res,np.zeros(shape=(N_steps-res.shape[0],)+res.shape[1:])),axis=0)
    
    #[timesteps,80,80,3]
    if not LSTM:
        #[80,80,3*timesteps]
        res = np.transpose(res,[1,2,3,0]).reshape(OBSERVATION_SPACE.shape)
    epi_steps +=1

    if env_obs.events.collisions or env_obs.events.reached_goal:
        states=deque(maxlen=N_steps)
        epi_steps = 0
    
    return np.array(res, dtype=np.float32)


# reward function
def reward_adapter(env_obs, env_reward):
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if env_obs.events.collisions else 0

    # if args.algo == 'value_penalty' or args.algo == 'policy_constraint':
    #     return goal + crash
    # else:
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


#### RL training ####
parser = Trainer.get_argument()
# parser.add_argument("algo", help="algorithm to run")
# parser.add_argument("scenario", help="scenario to run")
# parser.add_argument("--prior", help="path to the expert prior models", default=None)
args = parser.parse_args()
args.scenario = 'left_turn'
args.max_steps = 10e4
args.save_summary_interval = 128

args.show_progress = False
args.use_prioritized_rb = False
args.n_experiments = 1
args.test_episodes=10
args.test_interval=10000
args.gpu=0
# args.logdir = f'./train_results/{args.scenario}/{args.algo}'

# define scenario
if args.scenario == 'left_turn':
    scenario_path = ['scenarios/left_turn_new']
    max_episode_steps = 400
elif args.scenario == 'roundabout':
    scenario_path = ['scenarios/roundabout']
    max_episode_steps = 600
else:
    raise NotImplementedError

# define agent interface
agent_interface = AgentInterface(
    max_episode_steps=max_episode_steps,
    waypoints=True,
    neighborhood_vehicles=NeighborhoodVehicles(radius=60),
    rgb=RGB(80, 80, 32/80),
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

# if args.algo == 'gail':
#     expert_trajs = load_expert_trajectories(args.prior+'/*.npz')
print(bool(LSTM&bool((1-CNN_INPUTS))))
for i in range(args.n_experiments):
    print(f'Progress: {i+1}/{args.n_experiments}')

    # create env
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=i)
    env.observation_space = OBSERVATION_SPACE
    env.action_space = ACTION_SPACE
    env.agent_id = AGENT_ID

    test_env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: test_agent_spec}, headless=True, seed=i)
    test_env.observation_space = OBSERVATION_SPACE
    test_env.action_space = ACTION_SPACE
    test_env.agent_id = AGENT_ID
    policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                    batch_size=32, auto_alpha=True, memory_capacity=int(2e4),n_warmup=5000,lr=3e-4,
                    gpu=0,make_prediction=False,pred_trajs=pred_trajs,
                    actor_critic=GaussianActorCritic(
                        state_shape=env.observation_space.shape,
                        action_dim=2,
                        max_action=1.,
                        squash=True,
                        state_input=bool(1-CNN_INPUTS),
                        lstm=bool(LSTM&bool((1-CNN_INPUTS))),
                        cnn_lstm=False,
                        trans=False,
                        ego_surr=False,use_trans=True,neighbours=neighbors,time_step=N_steps,debug=False,bptt=False,hidden_activation="elu",
                        make_prediction=bool(future_steps>0),predict_trajs=pred_trajs
                    ),
                    actor_critic_target=GaussianActorCritic(
                        name='ac_target',
                        state_shape=env.observation_space.shape,
                        action_dim=2,
                        max_action=1.,
                        squash=True,
                        state_input=bool(1-CNN_INPUTS),
                        lstm=bool(LSTM&bool((1-CNN_INPUTS))),
                        cnn_lstm=False,
                        trans=False,
                        ego_surr=False,use_trans=True,neighbours=neighbors,time_step=N_steps,debug=False,bptt=False,hidden_activation="elu",
                        make_prediction=bool(future_steps>0),predict_trajs=pred_trajs
                    ),
                    )
    trainer = Trainer(policy=policy,env=env,args=args,test_env=test_env,save_path='sac_lstm',
                            use_mask=True,bptt_hidden=256,use_ego=False,make_predictions=future_steps)

    # begin training
    trainer()

    # close env
    env.close()