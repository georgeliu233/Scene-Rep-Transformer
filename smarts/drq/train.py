import copy
import math
import os
import pickle as pkl
import sys
import time

import numpy as np

import dmc2gym
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
import json
from tf2rl.experiments.trainer import Trainer
# from video import VideoRecorder

torch.backends.cudnn.benchmark = True

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB,Waypoints
from smarts.core.controllers import ActionSpaceType

ep_step = 0
def reward_adapter(env_obs, env_reward):
    global ep_step
    progress = env_obs.ego_vehicle_state.speed * 0.1
    goal = 1 if env_obs.events.reached_goal else 0
    crash = -1 if (env_obs.events.collisions) else 0

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

def observation_adapter(env_obs):
    return np.transpose(env_obs.top_down_rgb[1],(2,0,1)) / 255.0

parser = Trainer.get_argument()
args = parser.parse_args()
args.scenario = ''

args.n_experiments = 3
args.n_steps=3

AGENT_ID = 'Agent-007'
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=-1000.0, high=1000.0, shape=(3,80,80))
f = '../SMARTS/'
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

args.logdir = 'model_'+scenario_path[0].split('/')[-1] +'cnn'

def make_env(cfg, i):
    agent_interface = AgentInterface(
        max_episode_steps=max_episode_steps,
        waypoints=Waypoints(planned_path),
        neighborhood_vehicles=NeighborhoodVehicles(radius=None),
        action=ActionSpaceType.LaneWithContinuousSpeed,
        rgb=RGB(80, 80, 32/80)
    )

    # define agent specs
    agent_spec = AgentSpec(
        interface=agent_interface,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
        info_adapter=info_adapter,
    )

    env = utils.FrameStack(env, k=cfg.frame_stack)

    print(f'Progress: {i+1}/{args.n_experiments}')
    # create env
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec}, headless=True, seed=i)
    env.observation_space = OBSERVATION_SPACE
    env.action_space = ACTION_SPACE
    env.agent_id = AGENT_ID

    return env


class Workspace(object):
    def __init__(self, cfg, i):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg, i)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_video else None)
        self.step = 0
        self.trajs = []
    
    def record_traj(self):
        with open(self.work_dir + 'trajs.json','w',encoding='utf-8') as writer:
             writer.write(json.dumps(self.trajs,ensure_ascii=False,indent=4))
        self.trajs = []

    def store_trajs(self,traj,observations):
        traj.append([observations[0], observations[1]])

    def evaluate(self):
        average_episode_reward = 0
        avg_test_steps = 0
        success_time = 0
        col_time = 0
        stag_time = 0
        traj = []

        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            # self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            all_step = []
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                # self.video_recorder.record(self.env)
                self.store_trajs(traj, info.ego_vehicle_state.position)
                episode_reward += reward
                episode_step += 1
            
            self.trajs.append(traj)
            traj = []

            event = info
            if event.reached_goal:
                success_time +=1
                all_step.append(episode_step)
            if event.collisions !=[]:
                col_time +=1
            if event.reached_max_episode_steps:
                stag_time+=1

            average_episode_reward += episode_reward
            # self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        success_time /= self.cfg.num_eval_episodes
        col_time /= self.cfg.num_eval_episodes
        stag_time /= self.cfg.num_eval_episodes

        if success_time==0:
            m_s,s_s = 0,0
        else:
            m_s,s_s =np.mean(all_step) , np.std(all_step)

        self.record_traj()

        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/success_rate', success_time,
                        self.step)
        self.logger.log('eval/col_time', col_time,
                        self.step)
        self.logger.log('eval/stag_time', stag_time,
                        self.step)
        self.logger.log('eval/m_s', m_s,
                        self.step)
        self.logger.log('eval/s_s', s_s,
                        self.step)
        self.logger.dump(self.step)


    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        success = []
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))
                
                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                event = info
                if event.reached_goal:
                    success.append(1.0)
                else:
                    success.append(0)

                self.logger.log('train/episode', episode, self.step)
                self.logger.log('train/success_rate', np.mean(success[-50:]), self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml', strict=True)
def main(cfg):
    from train import Workspace as W
    for i in range(args.n_experiments):
        workspace = W(cfg, i)
        workspace.run()


if __name__ == '__main__':
    main()
