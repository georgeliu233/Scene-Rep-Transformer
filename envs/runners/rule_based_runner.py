import argparse
import os
from pathlib import Path
import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.env.hiway_env import HiWayEnv

from tqdm import tqdm,trange
import json


class RuleBasedAgent(Agent):
    def act(self, obs):
        return "keep_lane"


class RulebasedTrainer:
    def __init__(
        self,
        args,
        scenario,
        max_episode,
        ):

        agent_ids = [args.AGENT_ID]
        agent_specs = {
            agent_id: AgentSpec(
                interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=max_episode),
                agent_builder=RuleBasedAgent,
            )
            for agent_id in agent_ids
        }
        self.agents = {aid: agent_spec.build_agent() for aid, agent_spec in agent_specs.items()}
        self.env = HiWayEnv(scenarios=scenario, agent_specs=agent_specs, headless=True)
        self.agent_ids = agent_ids
        self.agent_id = args.AGENT_ID
        self.test_time = args.test_episodes
    
    def evaluate_policy_continuously(self):
        success_time = 0
        col_time = 0
        stag_time = 0
        all_step = []
        ego_data = []
        for _ in trange(self.test_time):
            observations = self.env.reset()
            done = False
            epi_step = 0
            epi_data=[]
            while not done:
                agent_ids = list(observations.keys())
                actions = {aid: self.agents[aid].act(observations[aid]) for aid in agent_ids}
                observations, _, dones,infos = self.env.step(actions)
                done = dones[self.agent_id]
                epi_step +=1
                info = infos[self.agent_id]
                obs_event = info['env_obs']
                ego = obs_event.ego_vehicle_state
                line = [ego.position[0],ego.position[1],ego.speed,float(ego.heading)]
                epi_data.append(line)

                if done:
                    ego_data.append(epi_data)
                    info = infos[self.agent_id]
                    event = info['env_obs'].events
                    if event.reached_goal:
                        success_time +=1
                    all_step.append(epi_step)
                    if event.collisions !=[]:
                        col_time +=1
                    if event.reached_max_episode_steps:
                        stag_time+=1

        s_r,c_r,stag = success_time/test_time , col_time/test_time , stag_time/test_time
        print(f'Scenario:{sc},Success_rate:{s_r},Collsion rate:{c_r},Stagnation:{stag}')
        self.env.close()

        
