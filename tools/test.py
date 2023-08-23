import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from configs.init_configs import get_argument, set_configs
from algos.ppo import PPO
from algos.sac import SAC
from copy import deepcopy
from glob import glob
import gym
# DrQ is combined in SAC codes

from envs.runners import on_policy_trainer, on_policy_trainer_carla, \
    off_policy_trainer, off_policy_trainer_carla, rule_based_runner

def testing(args):

    args, algo_params, runner_params = set_configs(args, test=True)

    ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    
    # initilize env
    if args.scenario =='carla':
        # carla env spec
        OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.N_steps, args.dim,))
        from envs.carla.carla_env import InterSection
        test_env = InterSection()
        test_env.observation_space = OBSERVATION_SPACE
        test_env.action_space = ACTION_SPACE
    else:
        #smarts env spec
        from smarts.env.hiway_env import HiWayEnv
        from smarts.core.agent import AgentSpec
        from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, RGB, Waypoints
        from smarts.core.controllers import ActionSpaceType

        from envs.smarts.obs_adapter import NeighbourObsAdapter, ObsAdapter
        from envs.smarts.env_adapters import action_adapter, observation_adapter, reward_adapter, info_adapter

        obs_adapter = NeighbourObsAdapter('map', args.N_steps, args.dim,
                                        args.neighbors, return_map=True, prediction=True,
                                        planned_path=args.planned_path) 
        test_obs_adapter = deepcopy(obs_adapter)                   

        f = '../envs/smarts/'
        if args.scenario == 'left_turn':
            scenario_path = [f+'smarts_scenarios/left_turn']
            max_episode_steps = 400
        elif args.scenario == 'roundabout':
            scenario_path = [f+'smarts_scenarios/roundabout']
            max_episode_steps = 1000
        elif args.scenario == 'cross':
            scenario_path = [f+'smarts_scenarios/cross']
            max_episode_steps = 600
        elif args.scenario == 'roundabout_easy':
            scenario_path = [f+'smarts_scenarios/roundabout_easy']
            max_episode_steps = 400
        elif args.scenario == 'roundabout_medium':
            scenario_path = [f+'smarts_scenarios/roundabout_medium']
            max_episode_steps = 600
        else:
            raise NotImplementedError
        
        agent_interface = AgentInterface(
            max_episode_steps=max_episode_steps,
            waypoints=Waypoints(args.planned_path),
            neighborhood_vehicles=True,
            rgb=RGB(80, 80, 32/80),
            action=ActionSpaceType.LaneWithContinuousSpeed,
        )

        if args.algo == 'drq':
            def drq_adapter(observation):
                obs =  obs_adapter.observation_adapter(observation)[0]
                return obs

        agent_spec = AgentSpec(
            interface=agent_interface,
            observation_adapter=drq_adapter if args.algo == 'drq' else observation_adapter,
            reward_adapter=reward_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
        )
        test_agent_spec = deepcopy(agent_spec)

        # to fetch neighbor states
        neighbor_spec = AgentInterface(
                    max_episode_steps=None,
                    action=ActionSpaceType.Lane,
                    waypoints=Waypoints(args.planned_path)
                    )
        test_neighbor_spec = deepcopy(neighbor_spec)
        
        test_env = HiWayEnv(scenarios=scenario_path, agent_specs={args.AGENT_ID: agent_spec}, headless=True)
        OBSERVATION_SPACE = obs_adapter.OBSERVATION_SPACE
        test_env.observation_space = OBSERVATION_SPACE
        test_env.action_space = ACTION_SPACE
        test_env.agent_id = args.AGENT_ID
    
    # load default model if None:
    if args.model_dir == None:
        args.model_dir = f'../data/{args.algo}/{args.scenario}/ckpt'
    
    # set algo and runner args:
    if args.algo == 'ppo':
        from algos.modules.actor_critic_policy import GaussianActorCritic
        policy = PPO(actor_critic=GaussianActorCritic(**algo_params['actor_critic_params']), **algo_params)
        runner_page = on_policy_trainer_carla if args.scenario == 'carla' else on_policy_trainer
        runner = runner_page.OnPolicyTrainer(policy=policy, env=test_env, args=args, test_env=test_env, **runner_params)
    elif args.algo == 'dt':
        raise NotImplementedError('have not incoporated, please visit DT for train & eval')
    elif args.algo == 'rb':
        runner = rule_based_runner.RulebasedTrainer(args, scenario_path, max_episode_steps)
    else:
        policy = SAC(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.high.size,
                        max_action=ACTION_SPACE.high[0], **algo_params)
        runner_page = off_policy_trainer_carla if args.scenario == 'carla' else off_policy_trainer
        if args.scenario == 'carla':
            runner = runner_page.Trainer(policy=policy, env=test_env, test_env=test_env, args=args, **runner_params)
        else:  
            runner = runner_page.Trainer(policy=policy, env=test_env, test_env=test_env, args=args, 
                                obs_adapter=obs_adapter.obs_adapter, test_obs_adapter=test_obs_adapter.obs_adapter,
                                neighbor_spec=neighbor_spec,test_neighbor_spec=test_neighbor_spec, **runner_params)
    
    runner.evaluate_policy_continuously()
    

if __name__=='__main__':
    parser = get_argument()
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.set_visible_devices(devices=gpus[args.gpu], device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu], True)

    testing(args)



