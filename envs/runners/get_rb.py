import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent,OnPolicyAgent
from tf2rl.envs.utils import is_discrete


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1, ]  # space.n
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))


def get_default_rb_dict(size, env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": get_space_size(env.observation_space)},
            "next_obs": {
                "shape": get_space_size(env.observation_space)},
            "act": {
                "shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}


def get_replay_buffer(policy, env, use_prioritized_rb=False,
                      use_nstep_rb=False, n_step=1, size=None,timesteps=0,bptt_hidden=0,
                      make_predictions=10,pred_dim=5,use_map=False,neighbors=5,path_length=50,
                      multi_selection=False,represent=False):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if size is not None:
        kwargs["size"] = size

    if timesteps>0:
        kwargs["env_dict"]['mask'] = {"shape":(timesteps,)}
        if not issubclass(type(policy), OnPolicyAgent):
            kwargs["env_dict"]['next_mask'] = {"shape":(timesteps,)}
    if bptt_hidden>0:
        kwargs["env_dict"]['hidden'] = {"shape":(bptt_hidden,)}
        if not issubclass(type(policy), OnPolicyAgent):
            kwargs["env_dict"]['next_hidden'] = {"shape":(bptt_hidden,)}
        # kwargs["env_dict"]['timestep'] = {}
    if make_predictions and use_map:
        kwargs["env_dict"]['ego'] = {"shape":(make_predictions,pred_dim)}
        kwargs["env_dict"]['ego_mask'] = {"shape":(make_predictions,)}
        if multi_selection:
            kwargs["env_dict"]['pred_ego'] = {"shape":(make_predictions,2)}
    if use_map:
        kwargs["env_dict"]['map_state'] = {"shape":(2*(neighbors+1),path_length,pred_dim)}
        kwargs["env_dict"]['next_map_state'] = {"shape":(2*(neighbors+1),path_length,pred_dim)}
    
    if represent:
        kwargs["env_dict"]['future_obs'] = {"shape":(make_predictions,)+get_space_size(env.observation_space)}
        if use_map:
            kwargs["env_dict"]['future_map_state'] = {"shape":(make_predictions,2*(neighbors+1),path_length,pred_dim)}
        kwargs["env_dict"]['future_action'] = {"shape":(make_predictions,)+get_space_size(env.action_space)}
    

    # on-policy policy
    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["size"] = policy.horizon
        kwargs["env_dict"].pop("next_obs")
        kwargs["env_dict"].pop("rew")
        # TODO: Remove done. Currently cannot remove because of cpprb implementation
        # kwargs["env_dict"].pop("done")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if is_discrete(env.action_space):
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        
        return ReplayBuffer(**kwargs)
    
    
    # N-step prioritized
    if use_prioritized_rb and use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return PrioritizedReplayBuffer(**kwargs)

    # prioritized
    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    # N-step
    if use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return ReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)