import os
import gym
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
tf.config.set_visible_devices(devices=gpus[-2], device_type='GPU')

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent

import sys
#sys.path.append('/home/haochen/SMARTS_test_TPDM/sac_model/sac_pic.py')
from on_policy_trainer import OnPolicyTrainer
from tf2rl.algos.ppo import PPO
from actor_critic_policy import GaussianActorCritic
agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.LanerWithSpeed, max_episode_steps=1000,neighborhood_vehicles=True),
    agent_builder=None
)
agent_specs={
    'Agent-007':agent_spec
}

state_input = False
LSTM = False
N_steps = 0
neighbours = 0
use_neighbors=False


parser = OnPolicyTrainer.get_argument()
parser = PPO.get_argument(parser)
args = parser.parse_args()

args.max_steps=100000

# args.test_episodes=20
# args.test_interval=2500
args.save_model_interval = 20000
args.test_interval=1e7
args.gpu=0
args.save_summary_interval=int(1e3)
args.normalize_obs=False


args.scenario = 'r'

if args.scenario == 'left_turn':
    scenario_path = ['scenarios/left_turn_new']
    max_episode_steps = 400
elif args.scenario == 'r':
    scenario_path = ['scenarios/roundabout']
    max_episode_steps = 900
elif args.scenario == 'cross':
    scenario_path = ['scenarios/double_merge/cross']
    max_episode_steps = 600
elif args.scenario == 're':
    scenario_path = ['scenarios/roundabout_easy']
    max_episode_steps = 400
elif args.scenario == 'rm':
    scenario_path = ['scenarios/roundabout_medium']
    max_episode_steps = 600
else:
    raise NotImplementedError
args.logdir = ''+'ppo_model_'+scenario_path[0].split('/')[-1]

policy = PPO(
    state_shape=(80,80,3),
    action_dim=2,
    is_discrete=False,
    state_input=state_input,
    lstm=LSTM,
    batch_size=32,
    horizon=512,
    n_epoch=10,
    lr_actor=5e-4,
    trans=False,
    use_schdule=True,
    final_steps=args.max_steps,
    final_lr=1e-5,
    entropy_coef=0.05, 
    vfunc_coef=0.5,
    gpu=-1,
    actor_critic=GaussianActorCritic(
        state_shape=(80,80,3),
        action_dim=2,
        max_action=1.,
        squash=True,
        state_input=False,
        lstm=False,
        cnn_lstm=False,
        trans=False,
        ego_surr=use_neighbors,use_trans=True,neighbours=neighbours,time_step=N_steps,debug=False
    )
)
test_mode=True

if test_mode:
    args.test_episodes=50
    args.model_dir=''
    env = gym.make("smarts.env:hiway-v0",
                        scenarios=scenario_path,
                        agent_specs=agent_specs,
                        headless=True,
                        seed=2)
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    env.observation_space = gym.spaces.Box(low=0,high=255,shape=(80,80,3), dtype=np.float32)
    trainer = OnPolicyTrainer(policy=policy,env=env,args=args,test_env=env,state_input=state_input,lstm=LSTM,n_steps=N_steps,
    ego_surr=use_neighbors,surr_vehicles=neighbours,save_name=f'test_ppo_{args.scenario}_{0}')
    trainer.evaluate_policy_continuously()
else:
    for i in range(3):
        env = gym.make("smarts.env:hiway-v0",
                        scenarios=scenario_path,
                        agent_specs=agent_specs,
                        headless=True,
                        seed=i)
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        env.observation_space = gym.spaces.Box(low=0,high=255,shape=(80,80,3), dtype=np.float32)

        trainer = OnPolicyTrainer(policy=policy,env=env,args=args,test_env=env,state_input=state_input,lstm=LSTM,n_steps=N_steps,
        ego_surr=use_neighbors,surr_vehicles=neighbours,save_name=f'ppo_{args.scenario}_{i}')
        trainer()
