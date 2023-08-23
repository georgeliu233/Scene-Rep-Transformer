import os
import time
import json
import math
import pickle

import numpy as np
import tensorflow as tf

from cpprb import ReplayBuffer
from collections import deque

from tqdm import trange

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete

class OnPolicyTrainer(Trainer):
    def __init__(self,ego_surr=False,surr_vehicles=5,n_step=3,skip_timestep=3,save_name='./ppo', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_log = []
        self.eval_log = []
        self.step_log = []
        self.test_step = []
        self.success_rate=[]
        self.train_success_rate=[]
        self.ego_surr=ego_surr
        self.surr_vehicles=surr_vehicles
        self.save_name=save_name
        self.n_steps= n_step
        self.skip_timestep = skip_timestep
        print('normalized:',self._normalize_obs)
    

    def __call__(self):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env,n_step=self.n_steps,use_nstep_rb=True,size=20000)

        kwargs_local_buf = get_default_rb_dict(
            size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}

        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        episode_steps = 0
        episode_return = 0
        success_log = [0]
        episode_start_time = time.time()
        total_steps = np.array(0, dtype=np.int32)
        n_epoisode = 0
        obs, _, _ = self._env.reset()

        tf.summary.experimental.set_step(total_steps)
        init_dis = 0
        while total_steps < self._max_steps:
            # Collect samples
            for _ in range(self._policy.horizon):
                if episode_steps % self.skip_timestep ==0: 
                    action, logp, val = self._policy.get_action_and_val(obs)
                
                next_obs, reward, done, info= self._env.step(action)
                next_obs = next_obs[0]
        
                episode_return += reward

                episode_steps += 1
                total_steps += 1
                done_flag = done
                
                self.local_buffer.add(
                    obs=obs, act=action, next_obs=next_obs,
                    rew=reward, done=done_flag, logp=logp, val=val)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
             
                    success_log.append(1 if info[0] else 0)
                    tf.summary.experimental.set_step(total_steps)
                    
                    self.finish_horizon()
                    obs,_,_ = self._env.reset()
                    init_dis = 0

                    n_epoisode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    success = np.sum(success_log[-20:]) / 20
                    self.train_success_rate.append(success)
                    self.logger.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} Success:{5: 5.2f} FPS: {4:5.2f}".format(
                            n_epoisode, int(total_steps), episode_steps, episode_return, fps,success))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    self.return_log.append(episode_return)
                    self.step_log.append(int(total_steps))
              
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps ,success_rate= self.evaluate_policy(total_steps)
                    self.eval_log.append(avg_test_return)
                    self.test_step.append(avg_test_steps)
                    self.success_rate.append(success_rate)
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes,success_rate))
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(
                        name="Common/average_test_episode_length", data=avg_test_steps)
                    self.writer.flush()

                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            self.finish_horizon(last_val=val)

            tf.summary.experimental.set_step(total_steps)

            # Train actor critic
            if self._policy.normalize_adv:
                samples = self.replay_buffer.get_all_transitions()
                self.mean_adv = np.mean(samples["adv"])
                self.std_adv = np.std(samples["adv"])
                # Update normalizer
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])
            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                for _ in range(self._policy.n_epoch):
                    if False:
                        self.slide_window_lstm()
                    else:
                        samples = self.replay_buffer._encode_sample(
                            np.random.permutation(self._policy.horizon))
                        if self._normalize_obs:
                            samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
                        if self._policy.normalize_adv:
                            adv = (samples["adv"] - self.mean_adv) / (self.std_adv + 1e-8)
                        else:
                            adv = samples["adv"]

                        if 'mask' not in samples.keys():

                            for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                                target = slice(idx * self._policy.batch_size,
                                            (idx + 1) * self._policy.batch_size)
                                self._policy.train(
                                    states=samples["obs"][target],
                                    actions=samples["act"][target],
                                    advantages=adv[target],
                                    logp_olds=samples["logp"][target],
                                    returns=samples["ret"][target])
                        else:                
                            for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                                target = slice(idx * self._policy.batch_size,
                                            (idx + 1) * self._policy.batch_size)
                                self._policy.train(
                                    states=samples["obs"][target],
                                    actions=samples["act"][target],
                                    advantages=adv[target],
                                    logp_olds=samples["logp"][target],
                                    returns=samples["ret"][target],
                                    mask=np.array(samples['mask'][target]))
        tf.summary.flush()

    def slide_window_lstm(self):
        samples =self.replay_buffer.get_all_transitions()
        new_sample = {
            'obs':[],
            'act':[],
            'adv':[],
            'logp':[],
            'ret':[]
        }
        for i in range(self._policy.horizon-self.n_steps):
            new_sample['obs'].append(samples['obs'][i:i+self.n_steps])
            new_sample['ret'].append(samples['ret'][i:i+self.n_steps])
            new_sample['act'].append(samples['act'][i:i+self.n_steps])
            new_sample['logp'].append(samples['logp'][i:i+self.n_steps])

            new_sample['adv'].append(samples['adv'][i:i+self.n_steps])
        
        if self._normalize_obs:
            obs = self._obs_normalizer(np.array(new_sample["obs"]), update=False)
        else:
            obs = np.array(new_sample['obs'])
        if self._policy.normalize_adv:
            adv = (np.array(new_sample["adv"]) - self.mean_adv) / (self.std_adv + 1e-8)
        else:
            adv = np.array(new_sample["adv"])
        
        for idx in range(int(obs.shape[0] / self._policy.batch_size)):
            target = slice(idx * self._policy.batch_size,
                            (idx + 1) * self._policy.batch_size)
           
            self._policy.train(
                states=obs[target],
                actions=np.array(new_sample['act'])[target],
                advantages=adv[target],
                logp_olds=np.array(new_sample["logp"])[target],
                returns=np.array(new_sample["ret"])[target])
        

    def finish_horizon(self, last_val=0):
        self.local_buffer.on_episode_end()
        
        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.local_buffer.clear()
        

    def evaluate_policy(self, total_steps=0,plot_map_mode=None):
        avg_test_return = 0.
        avg_test_steps = 0
        success_time = 0

        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        
        col_time = 0
        stag_time = 0
        ego_data=[]
        step_data=[]
        full_step =[]
        for i in trange(self._test_episodes):
            episode_return = 0.
            episode_time = 0
            frames = []
            obs,_,_ = self._test_env.reset()

            avg_test_steps += 1
            ep_step =0

            flag=False
            init_dis=0
            epi_state=[]
            
            for _ in range(self._episode_max_steps):
                if ep_step % self.skip_timestep ==0: 
                    act, _ = self._policy.get_action(np.expand_dims(obs,0), test=True)
         
                next_obs, reward, done, info = self._test_env.step(act[0])
                next_obs = next_obs[0]
                avg_test_steps += 1
                episode_time+=1
                ep_step += 1

                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
                        rew=reward, done=done)

                episode_return += reward
                obs = next_obs
                if done:      
                    obs,_,_ = self._test_env.reset()
                    if info[0]:
                        success_time+=1
                        full_step.append(ep_step)
                    if info[1]:
                        col_time +=1
                    if info[3]:
                        stag_time +=1
                    init_dis=0
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
        
        s_r,c_r,stag = success_time/self._test_episodes , col_time/self._test_episodes , stag_time/self._test_episodes

        print(f'mean_return:{avg_test_return / self._test_episodes}'
              f'success rate:{s_r}'
              f'collision rate:{c_r}'
              f'stagnation:{stag}'
              )

        return avg_test_return / self._test_episodes, avg_test_steps / success_time, success_time/self._test_episodes
