import os
import time
import json

import numpy as np
import tensorflow as tf
import csv
from utils import ZFilter
from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete

from utils import split_future

def get_future_rb_dict(size,_env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            'ego':{"shape":_env.observation_space.shape[2:]}
        }
    }
class OnPolicyTrainer(Trainer):
    def __init__(self,save_path='./ppo_log',use_mask=False,bptt_hidden=0,use_ego=False,
    make_predictions=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        self.make_predictions = make_predictions

        if self.use_mask:
            self.timesteps=self._env.observation_space.shape[0]
            print('using mask,timesteps:',self.timesteps)
        if self.bptt_hidden>0:
            print('using bptt,hidden size',self.bptt_hidden)
        if self.use_ego:
            print('using ego state..')
        if self.make_predictions>0:
            print('use predictions')

    def __call__(self):
        # Prepare buffer
        
        self.replay_buffer = get_replay_buffer(self._policy, self._env,timesteps=self.timesteps,bptt_hidden=self.bptt_hidden,
        make_predictions=self.make_predictions)
        self.reward_scaler = ZFilter(shape=())

        # Prepare local buffer
        kwargs_local_buf = get_default_rb_dict(size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        
        if self.use_mask:
            kwargs_local_buf["env_dict"]['mask'] = {"shape":(self.timesteps,)}
        if self.bptt_hidden>0:
            kwargs_local_buf["env_dict"]['hidden'] = {"shape":(self.bptt_hidden,)}
            # kwargs["env_dict"]['timestep'] = {}
        
        if self.use_ego:
            print(self._env.observation_space.shape[2:])
            kwargs_local_buf["env_dict"]['ego']={"shape":self._env.observation_space.shape[2:]}
        
        # if self.make_predictions>0:
        #     fut_kwargs = get_future_rb_dict(size=self._policy.horizon, env=self._env)
        #     self.ego_buffer = ReplayBuffer(**fut_kwargs)

        self.local_buffer = ReplayBuffer(**kwargs_local_buf)
        

        # set up variables
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        total_steps = np.array(0, dtype=np.int32)
        n_episode = 0
        success_log = [0]
        episode_returns = []
        best_train = -np.inf

        # reset env
        obs = self._env.reset()

        if self.use_ego:
            obs,ego = obs[self._env.agent_id]
        else:
            ego=None
            obs = obs[self._env.agent_id]

        if self.bptt_hidden>0:
            hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
        else:
            hidden,full_hidden=None,None

        # start interaction
        tf.summary.experimental.set_step(total_steps)

        while total_steps <= self._max_steps:
            ##### Collect samples #####
            for _ in range(self._policy.horizon):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)

                # get policy actions
                if self.use_mask:
                    num = np.clip(episode_steps+1,0,self.timesteps)
                    mask = np.array([1]*num +[0]*(self.timesteps - num))
                    mask = np.expand_dims(mask, axis=0)
                else:
                    mask=None
                
                if self.bptt_hidden>0:
                    act, logp, val,n_h = self._policy.get_action_and_val(obs,mask=mask,hidden=hidden)
                    # print(n_h.shape)
                else:
                    act, logp, val = self._policy.get_action_and_val(obs,mask=mask)

                # bound actions
                if not is_discrete(self._env.action_space):
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act

                # roll out a step
                next_obs, reward, done, info = self._env.step({self._env.agent_id: env_act})

                if self.use_ego:
                    next_obs,next_ego = next_obs[self._env.agent_id]
                else:
                    next_ego=None
                    next_obs = next_obs[self._env.agent_id]

                reward = reward[self._env.agent_id]
                done = done[self._env.agent_id]
                info = info[self._env.agent_id]

                episode_steps += 1
                total_steps += 1
                episode_return += reward

                done_flag = done
                if (hasattr(self._env, "_max_episode_steps") and episode_steps == self._env._max_episode_steps):
                    done_flag = False
                
                # add a sampled step to local buffer
                
                self.local_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=reward, done=done_flag, logp=logp, val=val,
                hidden=np.reshape(hidden,(-1)),mask=mask,ego=ego)
               

                obs = next_obs
                if self.use_ego:
                    ego = next_ego 
                if self.bptt_hidden>0:
                    hidden = np.expand_dims(full_hidden[(episode_steps-1)%self.timesteps],0)
                    # print(hidden.shape)
                    if episode_steps%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)

                    

                # add to training log
                # if total_steps % 5 == 0:
                #     success = np.sum(success_log[-20:]) / 20
                #     with open(self._output_dir+'/training_log.csv', 'a', newline='') as csvfile:
                #         writer = csv.writer(csvfile)
                #         writer.writerow([n_episode, total_steps, episode_returns[n_episode-1] if episode_returns else -1, success, episode_steps])

                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)
                    
                    # if the task is successful
                    # print(info)
                    success_log.append(1 if info.reached_goal else 0)
                    success = np.sum(success_log[-20:]) / 20

                    # end this eposide
                    self.finish_horizon()
                    obs = self._env.reset()
                    if self.use_ego:
                        obs,ego = obs[self._env.agent_id]
                    else:
                        ego=None
                        obs = obs[self._env.agent_id]
                    if self.bptt_hidden>0:
                        hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
                    else:
                        hidden,full_hidden=None,None

                    n_episode += 1
                    episode_returns.append(episode_return)
                    fps = episode_steps / (time.time() - episode_start_time)

                    self.return_log.append(episode_return)
                    self.step_log.append(int(total_steps))
                    self.train_success_rate.append(success)
                    with open('/home/haochen/TPDM_transformer/'+self.save_name+'.json','w',encoding='utf-8') as writer:
                        writer.write(json.dumps([self.return_log,self.step_log,self.train_success_rate],ensure_ascii=False,indent=4))

                    # log information
                    self.logger.info(
                        "Total Episode: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                            n_episode, int(total_steps), episode_steps, episode_return, fps))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    tf.summary.scalar(name='Common/training_success', data=success)

                    # reset variables
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                    # save policy model
                    # if n_episode > 20 and np.mean(episode_returns[-20:]) >= best_train:
                    #     best_train = np.mean(episode_returns[-20:])
                    #     self._policy.actor.network.save('{}/Model/Model_{}_{:.4f}.h5'.format(self._logdir, n_episode, best_train))

                # test the policy
                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps ,success_rate= self.evaluate_policy(total_steps)
                    self.eval_log.append(avg_test_return)
                    self.test_step.append(avg_test_steps)
                    self.test_success_rate.append(success_rate)
                    
                    with open('/home/haochen/TPDM_transformer/'+self.save_name+'_test.json','w',encoding='utf-8') as writer:
                        writer.write(json.dumps([self.eval_log,self.test_success_rate,self.test_step],ensure_ascii=False,indent=4))
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f},success rate:{3}, over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes,success_rate))
                    self.writer.flush()
                
                    obs = self._env.reset()
                    # obs = obs[self._env.agent_id]
                    if self.use_ego:
                        obs,ego = obs[self._env.agent_id]
                    else:
                        ego=None
                        obs = obs[self._env.agent_id]

                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.perf_counter()

                # save checkpoint
                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            self.finish_horizon(last_val=val)

            tf.summary.experimental.set_step(total_steps)

            ##### Train actor critic #####
            if self._policy.normalize_adv:
                samples = self.replay_buffer.get_all_transitions()
                mean_adv = np.mean(samples["adv"])
                std_adv = np.std(samples["adv"])
                # Update normalizer
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])

            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                al,cl,pl = [],[],[]
                for _ in range(self._policy.n_epoch):
                    samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))
                    
                    # normalize observation
                    if self._normalize_obs:
                        samples["obs"] = self._obs_normalizer(samples["obs"], update=False)

                    # normalize advantage
                    if self._policy.normalize_adv:
                        adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
                    else:
                        adv = samples["adv"]
                    
                    # train policy
                    
                    # c = samples["cell"] if self.bptt_hidden>0 else None

                    actor_loss , critic_loss , pred = [],[],[]

                    for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                        target = slice(idx * self._policy.batch_size, (idx + 1) * self._policy.batch_size)
                        m = samples["mask"][target] if self.use_mask else None
                        h = samples["hidden"][target] if self.bptt_hidden>0 else None
                        ego = samples['ego'][target] if self.make_predictions>0 else None
                        ego_mask = samples['ego_mask'][target] if self.make_predictions>0 else None

                        res = self._policy.train(
                            states=samples["obs"][target],
                            actions=samples["act"][target],
                            advantages=adv[target],
                            logp_olds=samples["logp"][target],
                            returns=samples["ret"][target],
                            mask=m,
                            hidden=h,
                            ego=ego,
                            ego_mask=ego_mask
                            )
                        
                        actor_loss.append(res[0]) , critic_loss.append(res[1]) , pred.append(res[2])
                    
                al.append(actor_loss)
                cl.append(critic_loss)
                pl.append(pred)
                print('losses:actor:{},critic:{},prediction:{}'.format(np.mean(al),np.mean(cl),np.mean(pl)))
                        

        tf.summary.flush()

    def finish_horizon(self, last_val=0):
        self.local_buffer.on_episode_end()

        samples = self.local_buffer._encode_sample(np.arange(self.local_buffer.get_stored_size()))
        
        # add the value of the last step if any
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        m = samples["mask"] if self.use_mask else np.zeros((0,))
        h = samples["hidden"] if self.bptt_hidden>0 else np.zeros((0,))
        ego = samples['ego'] if self.make_predictions>0 else np.zeros((0,))
        if self.make_predictions>0:
            ego,mask = split_future(ego,future_steps=self.make_predictions)
            # print(ego.shape,mask.shape)
        else:
            mask = np.zeros((0,))

        # c = samples["cell"] if self.bptt_hidden>0 else np.zeros((0,))

        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]),mask=m,
            hidden=h,ego=ego,ego_mask=mask)

        # clear local buffer
        self.local_buffer.clear()
    


    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        avg_test_steps = 0
        
        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, self._test_env, size=self._episode_max_steps)
        success_times = 0
        for i in range(self._test_episodes):
            episode_return = 0.
            obs = self._test_env.reset()
            if self.use_ego:
                obs,ego = obs[self._test_env.agent_id]
            else:
                ego=None
                obs = obs[self._test_env.agent_id]
            avg_test_steps += 1

            if self.bptt_hidden>0:
                hidden,full_hidden = np.zeros((1,self.bptt_hidden)),np.zeros((self.timesteps,self.bptt_hidden))
            else:
                hidden,full_hidden=None,None

            for j in range(self._episode_max_steps):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)

                if self.use_mask:
                    num = np.clip(j+1,0,self.timesteps)
                    mask = np.array([1]*num +[0]*(self.timesteps - num))
                    mask = np.expand_dims(mask, axis=0)
                else:
                    mask=None

                if self.bptt_hidden>0:
                    act, n_h = self._policy.get_action(obs,mask=mask,hidden=hidden, test=True)
                else:
                    act, _ = self._policy.get_action(obs,mask=mask, test=True)
                act = (act if is_discrete(self._env.action_space) else
                       np.clip(act, self._env.action_space.low, self._env.action_space.high))

                next_obs, reward, done, info = self._test_env.step({self._test_env.agent_id: act})
                if self.use_ego:
                    next_obs,next_ego = next_obs[self._test_env.agent_id]
                else:
                    next_ego=None
                    next_obs = next_obs[self._test_env.agent_id]
                reward = reward[self._test_env.agent_id]
                done = done[self._test_env.agent_id]
                avg_test_steps += 1

                if self.use_ego:
                    ego = next_ego

                if self._save_test_path:
                    replay_buffer.add(obs=obs, act=act, next_obs=next_obs, rew=reward, done=done)
                    
                episode_return += reward
                obs = next_obs

                if self.bptt_hidden>0:
                    hidden = np.expand_dims(full_hidden[j%self.timesteps],0)
                    if (j+1)%self.timesteps==0:
                        full_hidden = n_h.squeeze(axis=0)

                if done:
                    if info[self._test_env.agent_id].reached_goal:
                        success_times +=1
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes,success_times / self._test_episodes