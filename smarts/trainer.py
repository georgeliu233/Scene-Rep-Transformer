import os
import time
import logging
import argparse
import csv
import json
import pickle

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from copy import deepcopy

import random
from matplotlib import pyplot as plt
from time import sleep
from scipy.stats import norm
from tqdm import tqdm

from tf2rl.experiments.utils import save_path, frames_to_gif
from get_rb import get_replay_buffer
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.envs.normalizer import EmpiricalNormalizer
from tensorflow.keras.models import load_model
from collections import deque

from utils import decode_map_xml

# if tf.config.experimental.list_physical_devices('GPU'):
#     for cur_device in tf.config.experimental.list_physical_devices("GPU"):
#         print(cur_device)
#         tf.config.experimental.set_memory_growth(cur_device, enable=True)

class Trainer:
    def __init__(
            self,
            policy,
            env,
            args,
            test_env=None,
            save_path='./ppo_log',
            use_mask=False,
            bptt_hidden=0,
            use_ego=False,
            make_predictions=0,
            path_length=0,
            use_map=False,
            obs_adapter=None,
            test_obs_adapter=None,
            neighbor_spec=None,
            test_neighbor_spec=None,
            params=None,
            neighbors=0,
            multi_prediction=1,
            multi_selection=False,
            pred_future_state=False,
            skip_timestep=1,
            sep_train=False,
            max_train_episode=0,
            test_pca=False):

        
        self.save_name = save_path
        self.return_log = []
        self.eval_log = []
        self.step_log = []
        self.test_step = []

        self.train_success_rate=[]

        self.train_success_rate =[]
        self.test_success_rate=[]

        self.use_mask=use_mask
        self.timesteps=0
        self.bptt_hidden=bptt_hidden
        self.use_ego = use_ego
        self.use_map = use_map
        self.path_length=path_length

        self.params = params
        self.sep_train = sep_train

        self.make_predictions = make_predictions
        self.multi_prediction = multi_prediction
        self.multi_selection = multi_selection
        self.neighbors = neighbors

        self.pred_future_state = pred_future_state
        self.skip_timestep = skip_timestep

        self.max_train_episode = max_train_episode
        self.test_pca = test_pca

        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(Trainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix="{}_{}".format(self._policy.policy_name, args.dir_suffix))
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        # if evaluate the model
        if self.use_mask:
            self.timesteps=self._env.observation_space.shape[0]
            if self.timesteps==80:
                self.timesteps=int(self._env.observation_space.shape[-1]/3)
            print('using mask,timesteps:',self.timesteps)
        if self.bptt_hidden>0:
            print('using bptt,hidden size',self.bptt_hidden)
        if self.use_ego:
            print('using ego state..')
        if self.make_predictions>0:
            print('use predictions')
            if self.multi_prediction>1:
                print(f'prediction steps update:{self.multi_prediction}')
            if self.multi_selection:
                print('using multi selection...')
            if self.pred_future_state:
                print('representation learning..')

        if self.use_map:
            print('use_ego neighbor maps')
            self.obs_adapter = obs_adapter
            self.test_obs_adapter = test_obs_adapter
            self.neighbor_spec = neighbor_spec
            self.test_neighbor_spec  = test_neighbor_spec
            assert self.obs_adapter is not None
            assert self.neighbor_spec is not None
        
        if self.max_train_episode>0:
            print(f'Using episode mode instead, epi num:{self.max_train_episode}')

        if args.evaluate:
            assert args.model_dir is not None
        self._set_check_point(args.model_dir)

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()
        self.smarts = None
        if self.use_map:
            self.smarts = self._env._smarts
            self._test_smarts = self._test_env._smarts

    def _set_check_point(self, model_dir):
        # Save and restore model
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, directory=self._output_dir, max_to_keep=5)

        if model_dir is not None:
            if not os.path.isdir(model_dir):
                self._latest_path_ckpt = model_dir
            else:
                self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self._checkpoint.restore(self._latest_path_ckpt)
            self.logger.info("Restored {}".format(self._latest_path_ckpt))
    
    def process_neighbors(self,ids,test):
        # current_vehicles = smarts.vehicle_index.social_vehicle_ids(vehicle_types=frozenset({"car"}))
        if not test:
            self.smarts.attach_sensors_to_vehicles(self.neighbor_spec, ids)
            obs, _, _, dones = self.smarts.observe_from(ids)
            return obs
        else:
            self._test_smarts.attach_sensors_to_vehicles(self.test_neighbor_spec, ids)
            obs, _, _, dones = self._test_smarts.observe_from(ids)
            return obs
    
    def neighbor_obs(self,obs,test=False):
        if test:
            # if obs.keys()!=self._test_env.agent_id:
            #     print(obs.keys())
            obs = obs[self._test_env.agent_id]
        else:
            obs = obs[self._env.agent_id]
            
        neighbors_id = set([n.id for n in obs.neighborhood_vehicle_states])
        # print(neighbors_id)
        n_obs = self.process_neighbors(neighbors_id,test)

        if test:
            obs,ego,map_state = self.test_obs_adapter(obs,n_obs)
        else:
            obs,ego,map_state = self.obs_adapter(obs,n_obs)
        return obs,ego,map_state

    def plot_traj(self,traj,ego,total_steps):
        plt.figure()
        for i in range(traj.shape[0]):
            for j in range(traj.shape[1]):
                x,y = traj[i,j,:,0],traj[i,j,:,1]
                x,y = x[np.nonzero(x)],y[np.nonzero(y)]
                plt.plot(x,y,color='blue')
                plt.scatter(x,y,color='blue')
            x,y = ego[i,:,0],ego[i,:,1]
            x,y = x[np.nonzero(x)],y[np.nonzero(y)]
            plt.plot(x,y,color='red')
            plt.scatter(x,y,color='red')
        plt.savefig(f'./test_maps/{self.save_name}_traj_{total_steps}.png')

    def plot_curr_obs(self,samples,step):
        obs = samples["obs"]#(6,5,5)
        maps = samples['map_state']#(6,10,5)

        obss,mapss,ms,mms = self._policy.test_rotate(obs,maps)
        obs,maps,mask,mmask = obss[0],mapss[0],ms[0],mms[0]
        obs,maps= obs.numpy(),maps.numpy()
        plt.figure()
        ego_x,ego_y = obs[0,:,0],obs[0,:,1]
        plt.scatter(ego_x[np.nonzero(ego_x)],ego_y[np.nonzero(ego_y)],s=10,color='red')
        plt.plot(ego_x[np.nonzero(ego_x)],ego_y[np.nonzero(ego_y)],color='black')
        map_x,map_y = maps[0,:,0],maps[0,:,1]
        plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],s=10,color='red')
        plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='black')
        map_x,map_y = maps[1,:,0],maps[1,:,1]
        plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],s=10,color='red')
        plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='black')
        for j in range(2,2*(self.neighbors+1),2):
            ego_x,ego_y = obs[j//2,:,0],obs[j//2,:,1]
            
            plt.scatter(ego_x,ego_y)
            plt.plot(ego_x,ego_y)
            map_x,map_y = maps[j,:,0],maps[j,:,1]
            plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)])
            plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)])
            map_x,map_y = maps[j+1,:,0],maps[j+1,:,1]
            plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)])
            plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)])
            plt.scatter(ego_x[-1], ego_y[-1],color='blue',marker='*',s=80)
        ego_x,ego_y = obs[0,:,0],obs[0,:,1]
        plt.scatter(ego_x[-1], ego_y[-1],color='red',marker='*',s=80)
        plt.savefig('./wps/sac_map_{}.png'.format(step))

    def __call__(self):
        total_steps = 0
        frame_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0
        episode_returns = []
        success_log = [0]
        best_train = -np.inf

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step,timesteps=self.timesteps,bptt_hidden=self.bptt_hidden,
            make_predictions=self.make_predictions,use_map=self.use_map,path_length=self.path_length,neighbors=self.neighbors,
            multi_selection=self.multi_selection,represent=self.pred_future_state)
        
        if self.make_predictions>0:
            local_queue = deque(maxlen=self.make_predictions)
            ego_queue = deque(maxlen=self.make_predictions)
        
            if self.pred_future_state:
                fut_state_queue = deque(maxlen=self.make_predictions)
                fut_action_queue = deque(maxlen=self.make_predictions)
                if self.use_map:
                    fut_map_queue = deque(maxlen=self.make_predictions)

        obs = self._env.reset()

        if self.use_ego:
            obs,ego = obs[self._env.agent_id]
            map_s = None
        elif self.use_map:
            obs,ego,map_s = self.neighbor_obs(obs)
        else:
            ego=None
            obs = obs[self._env.agent_id]
            map_s = None

        if self.bptt_hidden>0:
            hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
        else:
            hidden,full_hidden,next_hidden=None,None,None

        r = 0
        b_s = 0
        while total_steps < self._max_steps or (self.max_train_episode>0 and n_episode < self.max_train_episode):

            if self.use_mask:
                num = np.clip(episode_steps+1,0,self.timesteps)
                mask = np.array([1]*num +[0]*(self.timesteps - num))
                mask = np.expand_dims(mask, axis=0)

                n_num = np.clip(episode_steps+2,0,self.timesteps)
                next_mask = np.array([1]*n_num +[0]*(self.timesteps - n_num))
                next_mask = np.expand_dims(next_mask, axis=0)
            else:
                mask=None
                next_mask=None

            if episode_steps%self.skip_timestep==0:
                if total_steps < self._policy.n_warmup:
                    action = self._env.action_space.sample()
                    if self.multi_selection:
                        pred_ego = map_s[0,:,:2]
                    else:
                        pred_ego =None
                else:
                    if self.bptt_hidden>0:
                        action,n_h = self._policy.get_action(obs,mask=mask,init_state=hidden,map_state=np.expand_dims(map_s,axis=0),test=False)
                    else:
                        if self.use_map:
                            # print('a')
                            # map_s = 
                            if self.multi_selection:
                                action,pred_ego = self._policy.get_action(obs,mask=mask,map_state=np.expand_dims(map_s,axis=0),test=False)
                            else:
                                action = self._policy.get_action(obs,mask=mask,map_state=np.expand_dims(map_s,axis=0),test=False)
                                pred_ego = None
                act = action

            next_obs, reward, done, info = self._env.step({self._env.agent_id: act})
            # print('obs',obs[0,:,:2],action)
            # print(act)
            if self.use_ego:
                next_obs,next_ego = next_obs[self._env.agent_id]
                next_map_s=None
            elif self.use_map:
                next_obs,next_ego,next_map_s = self.neighbor_obs(next_obs)
            else:
                next_ego=None
                next_map_s=None
                next_obs = next_obs[self._env.agent_id]
            # print(obs[0,:,:2],next_obs[0,:,:2],episode_steps)

            reward = reward[self._env.agent_id]
            done = done[self._env.agent_id]
            info = info[self._env.agent_id]
            
            if self._show_progress:
                obs_tensor = tf.expand_dims(obs, axis=0)
                # agent distribution
                agent_dist = self._policy.actor._compute_dist(obs_tensor)
            
            r += self._policy.discount**(b_s) * reward
            b_s +=1

            episode_steps += 1
            episode_return += reward
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            # if the episode is finished
            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                episode_steps == self._env._max_episode_steps):
                done_flag = False
            
            if (episode_steps%self.skip_timestep==0) or done or episode_steps == self._episode_max_steps:
                # print(obs[0,:,:2],next_obs[0,:,:2])
                reward = r
                r = 0
                b_s = 0
                frame_steps+= 1
            
            if self.bptt_hidden>0 :
                if total_steps > self._policy.n_warmup:
                    next_hidden = np.expand_dims(full_hidden[(episode_steps-1)%self.timesteps],0)
                    # print(hidden.shape)
                    if episode_steps%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)
                else:
                    next_hidden = hidden
            # print(episode_steps)
            if episode_steps%self.skip_timestep==0 or done or episode_steps == self._episode_max_steps:
                
                if self.make_predictions>0:
                    # print(len(list(local_queue)))
                    # assert ego is not None
                    line = [obs,action,next_obs,reward,done_flag,mask,hidden,next_mask,next_hidden,map_s,next_map_s,pred_ego,next_ego]
                    # print(action)
                    # print('obs-p',obs[0,:,:2],action)
                    
                    local_queue.append(line)
                    ego_queue.append(next_ego)
                    if self.pred_future_state:
                        # print(next_obs)
                        # print(next_obs.shape)
                        fut_state_queue.append(next_obs)
                        fut_action_queue.append(action)
                        if self.use_map:
                            fut_map_queue.append(next_map_s)
                        # if total_steps > self._policy.n_warmup:
                        #     print(map_s.shape)
                        #     print(episode_steps)
                    else:
                        fut_state_queue=None
                        fut_action_queue=None
                        fut_map_queue=None


                    if frame_steps>=self.make_predictions:
                        assert len(list(local_queue))==self.make_predictions
                        assert len(list(ego_queue))==self.make_predictions
                        [obs,action,next_obs,reward,done_flag,mask,hidden,next_mask,next_hidden,map_s,next_map_s,pred_ego,next_ego] = list(local_queue)[0]
                        # print(obs)
                        # ego_queue.popleft()    
                        #full egos trajs and full mask
                        if self.use_map:
                            egos = np.array(list(ego_queue))
                        else:
                            egos = None
                        # print(egos.shape)

                        if self.pred_future_state:
                            fut_action=np.array(list(fut_action_queue))
                            fut_state = np.array(list(fut_state_queue))
                            if self.use_map:
                                fut_map = np.array(list(fut_map_queue))
                            else:
                                fut_map = None
                        else:
                            fut_state=None
                            fut_map=None
                            fut_action=None

                        if self.use_map:
                            ego_mask = np.ones((self.make_predictions,))
                        else:
                            ego_mask = None

                        replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag,mask=mask,hidden=hidden,
                    next_mask=next_mask,next_hidden=next_hidden,ego=egos,ego_mask=ego_mask,map_state=map_s,next_map_state=next_map_s,
                    pred_ego=pred_ego,future_obs=fut_state,future_map_state=fut_map,future_action=fut_action)

                else:
                    ego_mask=None
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag,mask=mask,hidden=hidden,
                    next_mask=next_mask,next_hidden=next_hidden,ego=ego,ego_mask=ego_mask,map_state=map_s,next_map_state=next_map_s,
                    pred_ego=pred_ego,future_obs=None,future_map_state=None,future_action=None)

                obs = next_obs
                if self.use_ego:
                    ego = next_ego 
                if self.bptt_hidden>0:
                    hidden = next_hidden
                if self.use_map:
                    map_s = next_map_s
                    ego = next_ego 
            
              
            # end of a episode
            if done or episode_steps == self._episode_max_steps:
                # if task is successful
                success_log.append(1 if info.reached_goal else 0)

                #process the rest pairs and clear the queue:
                if self.make_predictions>0:
                    if self.use_map:
                        shape = list(ego_queue)[0].shape
                    if self.pred_future_state:
                        shape_a = list(fut_action_queue)[0].shape
                        shape_s = list(fut_state_queue)[0].shape
                        if self.use_map:
                            shape_m = list(fut_map_queue)[0].shape

                    for p in range(len(list(local_queue))):

                        [obs,action,next_obs,reward,done_flag,mask,hidden,next_mask,next_hidden,map_s,next_map_s,pred_ego,next_ego] =local_queue.popleft()
                        #full egos trajs and full mask
                        
                        if self.pred_future_state:
                            
                            # print(np.array(list(fut_action_queue))[0],action,done_flag)

                            if len(list(fut_state_queue))==0:
                                fut_state = np.zeros((self.make_predictions,)+shape_s)
                                fut_action = np.zeros((self.make_predictions,)+shape_a)
                                if self.use_map:
                                    fut_map = np.zeros((self.make_predictions,)+shape_m)
                            else:
                                fut_state = np.concatenate( (np.array(list(fut_state_queue)) , np.zeros((self.make_predictions - len(list(fut_state_queue)),)+shape_s) ),axis=0)
                                fut_action = np.concatenate( (np.array(list(fut_action_queue)) , np.zeros((self.make_predictions - len(list(fut_action_queue)),)+shape_a) ),axis=0)
                                if self.use_map:
                                    fut_map = np.concatenate( (np.array(list(fut_map_queue)) , np.zeros((self.make_predictions - len(list(fut_map_queue)),)+shape_m) ),axis=0)
                            
                            fut_state_queue.popleft()
                            fut_action_queue.popleft()
                            if self.use_map:
                                fut_map_queue.popleft()
                            assert len(list(local_queue))==len(list(fut_state_queue)),(len(list(local_queue)),len(list(fut_state_queue)))
                            assert len(list(local_queue))==len(list(fut_action_queue))
                            if self.use_map:
                                assert len(list(local_queue))==len(list(fut_map_queue))
                            else:
                                fut_map=None
                        else:
                            fut_state=None
                            fut_map=None
                            fut_action=None

                        if self.use_map:
                            if len(list(ego_queue))==0:
                                egos = np.zeros((self.make_predictions,)+shape)
                            else:
                                egos = np.concatenate( (np.array(list(ego_queue)) , np.zeros((self.make_predictions - len(list(ego_queue)),)+shape) ),axis=0)
                            # print(np.array(list(ego_queue)).shape)
                            ego_queue.popleft()
                            assert len(list(local_queue))==len(list(ego_queue))

                            ego_mask = [1]*len(list(ego_queue)) +[0]*(self.make_predictions - len(list(ego_queue)))
                        else:
                            egos,ego_mask = None,None

                        replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag,mask=mask,hidden=hidden,
                        next_mask=next_mask,next_hidden=next_hidden,ego=egos,ego_mask=ego_mask,map_state=map_s,next_map_state=next_map_s,
                        pred_ego=pred_ego,future_obs=fut_state,future_map_state=fut_map,future_action=fut_action)
                    
                    assert len(list(local_queue))==0
                    if self.use_map:
                        assert len(list(ego_queue))==0

                    if self.pred_future_state:
                        # fut_action_queue.clear()
                        
                        assert len(list(fut_state_queue))==0
                        assert len(list(fut_action_queue))==0
                        if self.use_map:
                            assert len(list(fut_map_queue))==0
                        
                replay_buffer.on_episode_end()
                obs = self._env.reset()
                if self.use_ego:
                    obs,ego = obs[self._env.agent_id]
                    map_s = None
                elif self.use_map:
                    obs,ego,map_s = self.neighbor_obs(obs)
                else:
                    ego=None
                    obs = obs[self._env.agent_id]
                    map_s = None

                if self.bptt_hidden>0:
                    hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
                else:
                    hidden,full_hidden,next_hidden=None,None,None
                
                # display info
                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                success = np.sum(success_log[-20:]) / 20

                self.return_log.append(episode_return)
                self.step_log.append(int(total_steps))
                self.train_success_rate.append(success)
                with open('./'+self.save_name+'.json','w',encoding='utf-8') as writer:
                    writer.write(json.dumps([self.params,self.return_log,self.step_log,self.train_success_rate],ensure_ascii=False,indent=4))

                self.logger.info("Total Episode: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} Success: {4: 5.2f} FPS:{5:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return,success ,fps))

                tf.summary.scalar(name="Common/training_return", data=episode_return)
                tf.summary.scalar(name='Common/training_success', data=success)
                tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)

                # reset variables
                episode_returns.append(episode_return)
                episode_steps = 0
                frame_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

                # save policy model
                if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
                    best_train = np.mean(episode_returns[-20:])
                    # self._policy.actor.network.save('{}/Model/Model_{}_{:.4f}.h5'.format(self._logdir, n_episode, best_train))

            if total_steps < self._policy.n_warmup:
                continue
            
            if self.multi_prediction>0 and self.make_predictions>0:
                p_l = []
                for _ in range(self.multi_prediction):
                    samples = replay_buffer.sample(self._policy.batch_size)
                    eg = samples['ego'] if self.make_predictions>0 else None
                    mp = samples['map_state'] if self.use_map else None
                    pred_loss,traj,eg = self._policy.pred_traj(samples["obs"],samples["act"], mp,eg)
                    p_l.append(pred_loss.numpy())
                if total_steps % 2000==0:
                    print(f'prediction_loss:{np.mean(p_l)}')
                    # print(eg.numpy()[0,:,:2])
                    self.plot_traj(traj.numpy(),eg.numpy(),total_steps)

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)

                m = samples["mask"] if self.use_mask else None
                nm = samples["next_mask"] if self.use_mask else None
                h = samples["hidden"] if self.bptt_hidden>0 else None
                nh = samples["next_hidden"] if self.bptt_hidden>0 else None

                eg = samples['ego'] if self.make_predictions>0 and self.use_map else None
                ego_m = samples['ego_mask'] if self.make_predictions>0 and self.use_map else None

                mp = samples['map_state'] if self.use_map else None
                n_mp = samples['next_map_state'] if self.use_map else None
                pd = samples['pred_ego'] if self.multi_selection else None

                f_a = samples['future_action'] if self.pred_future_state else None
                f_s = samples['future_obs'] if self.pred_future_state else None
                f_m = samples['future_map_state'] if self.pred_future_state and self.use_map else None

                if self.pred_future_state and self.sep_train:
                    simi_loss = self._policy.train_rep(state=samples["obs"], map_state=mp,
                     future_state=f_s, future_map_state=f_m, future_action=f_a)
                    if total_steps % 500==0:
                        print(f'similarity_loss:{np.mean(simi_loss)}')
                
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    _,pred_traj,pred_loss = self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        None if not self._use_prioritized_rb else samples["weights"],
                        mask=m,
                        hidden=h,
                        next_mask=nm,
                        next_hidden=nh,
                        ego=eg,
                        ego_mask=ego_m,
                        map_state=mp,
                        next_map_state=n_mp,
                        hist_traj=pd,
                        future_state=f_s, future_map_state=f_m, future_action=f_a
                        )
                    # if total_steps % 1000==0 and self.pred_future_state and not self.sep_train:
                    #     print(f'similarity_loss:{np.mean(pred_loss.numpy())/self.make_predictions}')
                    if total_steps%5000==0:
                        pass
                        # self.plot_curr_obs(samples,total_steps)
                    
                        
                    if False:#total_steps%2000==0 and self.make_predictions:
                        print('prediction loss: {}'.format(pred_loss))
                        plt.figure()
                        for j in range(eg.shape[0]):
                            length = int(np.sum(ego_m[j]))
                            if length<=1:
                                continue
                            plt.plot(eg[j,:length-1,0],eg[j,:length-1,1])
                            plt.scatter(eg[j,:,0],eg[j,:,1],color='black')
                            for q in range(pred_traj.shape[1]):
                                plt.plot(pred_traj[j,q,:length-1,0],pred_traj[j,q,:length-1,1])
                        # plt.savefig('/home/haochen/TPDM_transformer/test_maps/{}_traj_{}.png'.format(self.save_name,total_steps))

                if self._use_prioritized_rb:
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"], np.array(samples["done"], dtype=np.float32),
                        mask=m,
                        hidden=h,
                        next_mask=nm,
                        next_hidden=nh,
                        ego=eg,
                        ego_mask=ego_m,
                        map_state=mp,
                        next_map_state=n_mp)
                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_error) + 1e-6)
                    # tf.summary.scalar(name=self._policy.policy_name + "/td_error", data=tf.reduce_mean(td_error))

            if total_steps % self._test_interval == 0:
                avg_test_return, avg_test_steps,success_rate = self.evaluate_policy(total_steps)
                self.eval_log.append(avg_test_return)
                self.test_step.append(avg_test_steps)
                self.test_success_rate.append(success_rate)

                with open('./'+self.save_name+'_test.json','w',encoding='utf-8') as writer:
                    writer.write(json.dumps([self.eval_log,self.test_success_rate,self.test_step],ensure_ascii=False,indent=4))
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes,success_rate))
                


                tf.summary.scalar(name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(name="Common/average_test_episode_length", data=avg_test_steps)
                tf.summary.scalar(name="Common/fps", data=fps)
                self.writer.flush()
                
                # reset env
                obs = self._env.reset()
                if self.use_ego:
                    obs,ego = obs[self._env.agent_id]
                    map_s = None
                elif self.use_map:
                    obs,ego,map_s = self.neighbor_obs(obs)
                else:
                    ego=None
                    obs = obs[self._env.agent_id]
                    map_s = None

                if self.bptt_hidden>0:
                    hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
                else:
                    hidden,full_hidden,next_hidden=None,None,None

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            # save checkpoint
            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()

        tf.summary.flush()

    def evaluate_policy_continuously(self,plot_map_mode=False,map_dir=''):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        # self.evaluate_policy(total_steps=0)
        # while True:
        if plot_map_mode:
            print('using map plot')
            xml_dir = map_dir+'map.net.xml'
            self.xml_dir = xml_dir
            print(xml_dir)
            sc = map_dir.split('/')[-2]
            name = f'./res_plt_{sc}/'
            print(name)
            if not os.path.exists(name):
                os.makedirs(name)
            self.plot_name = name

        res = self.evaluate_policy(total_steps=0,plot_map_mode=plot_map_mode)
        print("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                    res[0], res[1], self._test_episodes,res[2]))
    
    def plot_his_trajs(self,obs,maps,epi,step,val):
        plt = decode_map_xml(self.xml_dir)
        ego_x,ego_y = obs[0,:,0],obs[0,:,1]
        
        plt.xlim([ego_x[-1]-50,ego_x[-1]+50])
        plt.ylim([ego_y[-1]-50,ego_y[-1]+50])
        plt.scatter(ego_x[np.nonzero(ego_x)],ego_y[np.nonzero(ego_y)],s=10,color='red')
        plt.plot(ego_x[np.nonzero(ego_x)],ego_y[np.nonzero(ego_y)],color='black')
        plt.scatter(ego_x[-1], ego_y[-1],color='red',marker='*',s=100,label='ego_hist')
        map_x,map_y = maps[0,:,0],maps[0,:,1]
        v = val[0]

        plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],c=v[np.nonzero(map_y)],cmap='Blues',label='ego_map')
        plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='darkblue')
        map_x,map_y = maps[1,:,0],maps[1,:,1]
        v = val[1]
        plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],c=v[np.nonzero(map_y)],cmap='Blues')
        plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='darkblue',alpha=0.2)


        for j in range(2,2*(self.neighbors+1),2):
            ego_x,ego_y = obs[j//2,:,0],obs[j//2,:,1]
            label_n = 'neighbor_hist' if j==2 else None
            label_m = 'neighbor_map' if j==2 else None
            plt.plot(ego_x[np.nonzero(ego_x)],ego_y[np.nonzero(ego_x)],color='darkcyan')
            
            map_x,map_y = maps[j,:,0],maps[j,:,1]
            v = val[j]
            plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],c=v[np.nonzero(map_y)],cmap='OrRd',label=label_m)
            plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='darkorange',alpha=0.2)

            map_x,map_y = maps[j+1,:,0],maps[j+1,:,1]
            v = val[j+1]
            plt.scatter(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],c=v[np.nonzero(map_y)],cmap='OrRd')
            plt.plot(map_x[np.nonzero(map_x)],map_y[np.nonzero(map_y)],color='darkorange',alpha=0.2)

            plt.scatter(ego_x[-1], ego_y[-1],color='darkcyan',marker='*',s=100,label=label_n)
            

        # ego_x,ego_y = obs[0,:,0],obs[0,:,1]
        # plt.scatter(ego_x[-1], ego_y[-1],color='red',marker='*',s=80)
        plt.legend()
        plt.savefig(self.plot_name+f'map_{epi}_{step}.png')

    def evaluate_policy(self, total_steps,plot_map_mode=False):
        tf.summary.experimental.set_step(total_steps)
        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())

        avg_test_return = 0.
        avg_test_steps = 0

        success_time = 0
        col_time = 0
        stag_time = 0
        ego_data=[]
        all_step=[]
        full_step =[]

        pca_x=[]
        pca_y=[]

        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size=self._episode_max_steps)

        for i in tqdm(range(self._test_episodes)):
            episode_return = 0.
            epi_step = 0
            epi_state=[]

            obs = self._test_env.reset()
            if self.use_ego:
                obs,ego = obs[self._test_env.agent_id]
                map_s = None
            elif self.use_map:
                obs,ego,map_s = self.neighbor_obs(obs,test=True)
            else:
                ego=None
                obs = obs[self._test_env.agent_id]
                map_s = None

            if self.bptt_hidden>0:
                hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
            else:
                hidden,full_hidden,next_hidden=None,None,None
            avg_test_steps += 1

            for j in range(self._episode_max_steps):
                if self.use_mask:
                    num = np.clip(j+1,0,self.timesteps)
                    mask = np.array([1]*num +[0]*(self.timesteps - num))
                    mask = np.expand_dims(mask, axis=0)
                else:
                    mask=None

                if epi_step%self.skip_timestep==0:
                    if plot_map_mode:
                        h,val = self._policy.get_hidden_and_val(obs,map_state=np.expand_dims(map_s,0),test=True)
                        self.plot_his_trajs(obs,map_s,i,j,val)
                    # print(obs[0,:,:2])
                    if self.bptt_hidden>0:
                        action,n_h = self._policy.get_action(obs,mask=mask,init_state=hidden,test=True)
                    else:
                        action = self._policy.get_action(obs,test=True,mask=mask,map_state=np.expand_dims(map_s,0))
                    
                    if self.test_pca:
                        hs_a,q = self._policy.get_pca_val(obs,action,map_state=np.expand_dims(map_s,0),test=True)
                        pca_x.append(hs_a)
                        pca_y.append(q)

                    act = action

                # print(action)
                
                next_obs, reward, done, info = self._test_env.step({self._test_env.agent_id: act})
                obs_event = next_obs[self._test_env.agent_id]
                ego = obs_event.ego_vehicle_state
                line = [ego.position[0],ego.position[1],ego.speed,float(ego.heading)]
                epi_state.append(line)
                
                if self.use_ego:
                    next_obs,next_ego = next_obs[self._test_env.agent_id]
                    next_map_s=None
                elif self.use_map:
                    next_obs,next_ego,next_map_s = self.neighbor_obs(next_obs,test=True)
                else:
                    next_ego=None
                    next_map_s=None
                    next_obs = next_obs[self._test_env.agent_id]

                reward = reward[self._test_env.agent_id]
                done = done[self._test_env.agent_id]
                info = info[self._test_env.agent_id]
                
                # print(info)
                

                avg_test_steps += 1
                epi_step += 1

                # if epi_step%self.skip_timestep!=0 and (not done):
                #     continue
                
                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done)
                
                if self.bptt_hidden>0 :
                    next_hidden = np.expand_dims(full_hidden[(j)%self.timesteps],0)
                    # print(hidden.shape)
                    if (j+1)%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)

                episode_return += reward
                obs = next_obs

                if self.use_ego:
                    ego = next_ego 
                if self.use_map:
                    map_s = next_map_s
                if self.bptt_hidden>0:
                    hidden = next_hidden
                
                if done:
                    ego_data.append(epi_state)
                    event = info
                    if event.reached_goal:
                        success_time +=1
                        all_step.append(epi_step)
                    if event.collisions !=[]:
                        col_time +=1
                    if event.reached_max_episode_steps:
                        stag_time+=1
                    print(success_time,col_time,event)
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            avg_test_return += episode_return
        
        s_r,c_r,stag = success_time/self._test_episodes , col_time/self._test_episodes , stag_time/self._test_episodes
        if success_time==0:
            m_s,s_s = 0,0
        else:
            m_s,s_s =np.mean(all_step) , np.std(all_step)
        
        print(f'mean_return:{avg_test_return / self._test_episodes}'
              f'success rate:{s_r}'
              f'collision rate:{c_r}'
              f'stagnation:{stag}'
              f'step:{m_s},{s_s}'
              )
            
        if self.test_pca:
            with open(''+self.save_name+'_test.pkl','wb') as writer:
                pickle.dump([pca_x,pca_y], writer)

        with open(''+self.save_name+'_test.json','w',encoding='utf-8') as writer:
            writer.write(json.dumps([s_r,c_r,stag,m_s,s_s,ego_data],ensure_ascii=False,indent=4))

        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes,success_time / self._test_episodes

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(5e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true', default=False,
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(20e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=20,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        parser.add_argument('--scenario',default='')
        
        return parser
