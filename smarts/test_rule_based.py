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


def parse_args():
    parser = argparse.ArgumentParser("Rule based runner")
    return parser.parse_args()


def main(sc,scenario,max_episode,test_time):
    
    agent_ids = ['007']
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=max_episode),
            agent_builder=RuleBasedAgent,
        )
        for agent_id in agent_ids
    }

    agents = {aid: agent_spec.build_agent() for aid, agent_spec in agent_specs.items()}

    env = HiWayEnv(scenarios=scenario, agent_specs=agent_specs)

    success_time = 0
    col_time = 0
    stag_time = 0
    all_step = []
    ego_data = []
    for _ in trange(test_time):
        observations = env.reset()
        done = False
        epi_step = 0
        epi_data=[]
        while not done:
            agent_ids = list(observations.keys())
            actions = {aid: agents[aid].act(observations[aid]) for aid in agent_ids}
            observations, _, dones,infos = env.step(actions)
            done = dones["007"]
            epi_step +=1
            info = infos['007']
            obs_event = info['env_obs']
            ego = obs_event.ego_vehicle_state
            line = [ego.position[0],ego.position[1],ego.speed,float(ego.heading)]
            epi_data.append(line)

            if done:
                ego_data.append(epi_data)
                info = infos["007"]
                event = info['env_obs'].events
                if event.reached_goal:
                    success_time +=1
                all_step.append(epi_step)
                if event.collisions !=[]:
                    col_time +=1
                if event.reached_max_episode_steps:
                    stag_time+=1
    s_r,c_r,stag = success_time/test_time , col_time/test_time , stag_time/test_time
    print(f'Scenario:{sc},Success_rate:{s_r},Collsion rate:{c_r},Stagnation:{stag},avg_time{np.mean(all_step)},{np.std(all_step)}')
    sc_name = scenario[0].split('/')[-1]
    with open(f'./test_results/rb_{sc_name}_test.json','w',encoding='utf-8') as writer:
        writer.write(json.dumps([s_r,c_r,stag,ego_data],ensure_ascii=False,indent=4))
    env.close()


if __name__ == "__main__":
    args = parse_args()
    sc_list = ['left_turn','cross','re','rm','r']
    for sc in sc_list:
        args.scenario = sc
        if args.scenario == 'left_turn':
            scenario_path = ['scenarios/left_turn_new']
            max_episode_steps = 400
        elif args.scenario == 'r':
            scenario_path = ['scenarios/roundabout']
            max_episode_steps = 900
        elif args.scenario == 'cross':
            scenario_path = ['scenarios/double_merge/cross_test']
            max_episode_steps = 600
        elif args.scenario == 're':
            scenario_path = ['scenarios/roundabout_easy']
            max_episode_steps = 400
        elif args.scenario == 'rm':
            scenario_path = ['scenarios/roundabout_medium']
            max_episode_steps = 600
        else:
            raise NotImplementedError
        main(sc,scenario_path,max_episode_steps,test_time=50)