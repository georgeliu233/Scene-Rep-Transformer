
from ppo import PPO

from tf2rl.experiments.trainer import Trainer
from org_onpolicy_trainer import OnPolicyTrainer

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
from actor_critic_policy import GaussianActorCritic
from collections import deque
from utils import NeighbourAgentBuffer

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.set_visible_devices(devices=gpus[-1], device_type='GPU')



#### Environment specs ####
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

AGENT_ID = 'Agent-007'


LSTM=False
CNN_INPUTS=False
N_steps=10
future_steps=10
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

states = deque(maxlen=N_steps)
epi_steps = 0
test_states = deque(maxlen=N_steps)
test_epi_steps = 0

neighbours_buffer = NeighbourAgentBuffer(state_shape=OBSERVATION_SPACE.shape,hist_length=N_steps,future_length=10,query_mode='history_only')
test_neighbours_buffer = NeighbourAgentBuffer(state_shape=OBSERVATION_SPACE.shape,hist_length=N_steps,future_length=10,query_mode='history_only')

def neighbor_adapter(obs):
    global neighbours_buffer
    global states
    global epi_steps

    ego= obs.ego_vehicle_state
    neighbours = obs.neighborhood_vehicle_states
    
    #x,y,psi
    ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
    
    states.append(ego_state)
    
    dis_list = []
    min_ind = []
    id_list = []
    # print(epi_steps)
    if len(neighbours)>0:  
        for neighbour in neighbours:
            x,y= neighbour.position[0],neighbour.position[1]
            psi = float(neighbour.heading)
            speed = neighbour.speed
            # print(x,y,ego_state[0],ego_state[1],dis)
            dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
            neighbours_buffer.add(ids=neighbour.id, values=[x,y,speed,dis,psi],timesteps=epi_steps)
            dis_list.append(dis)
            id_list.append(neighbour.id)
        
        min_ind = np.argsort(dis_list)

        n_id_list = [id_list[i] for i in min_ind]
        id_list = n_id_list

    if len(id_list)>0:
        neighbors_state = neighbours_buffer.query_neighbours(curr_timestep=epi_steps,curr_ids=id_list
        ,keep_top=neighbors,pad_length=min(epi_steps+1,N_steps))
    else:
        neighbors_state = np.zeros((neighbors,OBSERVATION_SPACE.shape[1],OBSERVATION_SPACE.shape[2]))
    
    # if len(np.array(neighbors_state).shape)==2:
    #     print(neighbors_state)

    new_states =np.concatenate((np.expand_dims(list(states),0),neighbors_state),axis=0)
    
    if new_states.shape[1]<N_steps:
        padded = np.zeros((new_states.shape[0],N_steps-new_states.shape[1],new_states.shape[2]))
        new_states = np.concatenate((new_states,padded),axis=1)

    epi_steps +=1   

    if obs.events.collisions or obs.events.reached_goal:
        states=deque(maxlen=N_steps)
        epi_steps = 0
        neighbours_buffer.clear()
    
    return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32)

def test_neighbor_adapter(obs):
    global test_neighbours_buffer
    global test_states
    global test_epi_steps

    ego= obs.ego_vehicle_state
    neighbours = obs.neighborhood_vehicle_states
    
    #x,y,psi
    ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
    
    test_states.append(ego_state)
    # print(ego_state)
    # print(len(neighbours),test_epi_steps)
    dis_list = []
    min_ind = []
    id_list = []
    # print(epi_steps)
    if len(neighbours)>0:  
        for neighbour in neighbours:
            x,y= neighbour.position[0],neighbour.position[1]
            psi = float(neighbour.heading)
            speed = neighbour.speed
            # print(x,y,ego_state[0],ego_state[1],dis)
            dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
            test_neighbours_buffer.add(ids=neighbour.id, values=[x,y,speed,dis,psi],timesteps=test_epi_steps)
            dis_list.append(dis)
            id_list.append(neighbour.id)
        
        min_ind = np.argsort(dis_list)

        n_id_list = [id_list[i] for i in min_ind]
        id_list = n_id_list

    if len(id_list)>0:
        neighbors_state = test_neighbours_buffer.query_neighbours(curr_timestep=test_epi_steps,curr_ids=id_list
        ,keep_top=neighbors,pad_length=min(test_epi_steps+1,N_steps))
    else:
        neighbors_state = np.zeros((neighbors,)+OBSERVATION_SPACE.shape[1:])

    new_states =np.concatenate((np.expand_dims(list(test_states),0),neighbors_state),axis=0)
    
    if new_states.shape[1]<N_steps:
        padded = np.zeros((new_states.shape[0],N_steps-new_states.shape[1],new_states.shape[2]))
        new_states = np.concatenate((new_states,padded),axis=1)

    test_epi_steps +=1   

    if obs.events.collisions or obs.events.reached_goal:
        test_states=deque(maxlen=N_steps)
        test_epi_steps = 0
        test_neighbours_buffer.clear()
    
    return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32)

# observation space
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
        res = np.transpose(res,[1,2,3,0]).reshape((80,80,3*N_steps))
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
    return 0.01 * progress + goal + crash

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
args.max_steps = 20e4
args.save_summary_interval = 128

args.show_progress = False
args.use_prioritized_rb = False
args.n_experiments = 1
args.test_episodes=20
args.test_interval=2500
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
    observation_adapter=neighbor_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

test_agent_spec = AgentSpec(
    interface=agent_interface,
    observation_adapter=test_neighbor_adapter,
    reward_adapter=reward_adapter,
    action_adapter=action_adapter,
    info_adapter=info_adapter,
)

# if args.algo == 'gail':
#     expert_trajs = load_expert_trajectories(args.prior+'/*.npz')

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

    policy = PPO(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size, max_action=ACTION_SPACE.high[0],
                    batch_size=32, clip_ratio=0.2, n_epoch=8, entropy_coef=0.01, horizon=512,lr_actor=5e-4,
                    lr_critic=5e-4,is_discrete=False,gpu=0,make_prediction=True,pred_trajs=pred_trajs,
                    actor_critic=GaussianActorCritic(
                        state_shape=env.observation_space.shape,
                        action_dim=2,
                        max_action=1.,
                        squash=True,
                        state_input=bool(~CNN_INPUTS),
                        lstm=bool(LSTM&(~CNN_INPUTS)),
                        cnn_lstm=False,
                        trans=False,
                        ego_surr=True,use_trans=True,neighbours=neighbors,time_step=N_steps,debug=False,bptt=True,hidden_activation="elu",
                        make_prediction=bool(future_steps>0),predict_trajs=pred_trajs,share_policy=True
                    ))
    trainer = OnPolicyTrainer(policy=policy,env=env,args=args,test_env=test_env,save_path='ppo_ac_trans_pred_2',
                            use_mask=True,bptt_hidden=256,use_ego=True,make_predictions=future_steps)

    # begin training
    trainer()

    # close env
    env.close()