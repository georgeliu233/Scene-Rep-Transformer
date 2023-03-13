from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec,Agent
from smarts.core.agent_interface import NeighborhoodVehicles, RGB,RoadWaypoints,Waypoints
from smarts.core.controllers import ActionSpaceType

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(AgentType.Laner,
    max_episode_steps=1000,neighborhood_vehicles=NeighborhoodVehicles(50.0),
    waypoints=Waypoints(50)
    ),
    agent_builder=None
)
agent_specs={
    'Agent-LHC':agent_spec
}
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=["scenarios/left_turn_new"],
    agent_specs=agent_specs,
    headless=True
)
neighbor_spec = AgentInterface(
            max_episode_steps=None,
            action=ActionSpaceType.Lane,
            road_waypoints=RoadWaypoints(10),
            waypoints=Waypoints(50)
        )
AGENT_ID = 'Agent-LHC'
# obs = env.reset()

def get_state(env_obs):
    ego = env_obs.ego_vehicle_state
    neighborhood_vehicles = env_obs.neighborhood_vehicle_states
    road_wp = env_obs.road_waypoints
    path = env_obs.waypoint_paths
    cloud = env_obs.lidar_point_cloud
    print(dir(ego))
    print(float(ego.heading))
    print(len(neighborhood_vehicles))
    print(neighborhood_vehicles[0])
    plt.figure()
    plt.scatter(ego.position[0], ego.position[1],marker='*')
    for n in neighborhood_vehicles:
        plt.scatter(n.position[0],n.position[1],marker='*',color='red')
    # print(road_wp)
    # print(path)
    # print(cloud)
    # cloud_printer(cloud)
    # print(road_wp)
    # print(road_wp)
    # print(road_wp)
    # print(road_wp.lanes.keys())
    # print(road_wp.route_waypoints)
    # waypoint_printer(road_wp)
    # print(path)
    # path_printer(path)
    # waypoint_printer(road_wp)

def process_neighbors(env,ids):
    smarts = env._smarts
    # current_vehicles = smarts.vehicle_index.social_vehicle_ids(
    #         vehicle_types=frozenset({"car"})
    #     )
    # print(current_vehicles)
    smarts.attach_sensors_to_vehicles(
        AgentSpec(
        interface=neighbor_spec,
        agent_builder=None,
        observation_adapter=None,
    )
        , ids)
    obs, _, _, dones = smarts.observe_from(ids)
    # print(dones)
    return obs
def test_wps():
    # print(env._smarts)
    obs = env.reset()
    obs = obs[AGENT_ID]
    # ids = obs.ego_vehicle_state
    # neighbors_id = []
    # for n in obs.neighborhood_vehicle_states:
    #     if isinstance(n.id,bytes):
    #         neighbors_id.append(str(n.id,'utf-8'))
    #     else:
    #         neighbors_id.append(n.id)
    # neighbors_id = set(neighbors_id)
    # # print(ids)
    # n_obs = process_neighbors(env,neighbors_id)
    # print(n_obs)
    i = 0
    
    while True:
        neighbors_id = set([n.id for n in obs.neighborhood_vehicle_states])
        n_obs = process_neighbors(env,neighbors_id)

        waypoint_printer(n_obs,i,obs)
        i+=1
        print(i)
        # break
        new_obs,r,done,info = env.step(
            {AGENT_ID: 'keep_lane'}
        )
        # buffer.add(obs=obs[0],ego=obs[1])
        obs = new_obs[AGENT_ID]

        if done[AGENT_ID]:
            break
    
def test_str():
    a = b'000000000000$car_type_3-flow-route-edge-east-EN_1_random-edge-south-NS_0_random-8222204009439192561--7502074924325504014--9-2.0'
    print(isinstance(a,bytes))
    b = str(a,'utf-8')
    print(b)

def cloud_printer(obs,i=0):
    cloud = obs.lidar_point_cloud
    cloud_info,hit,ray = cloud
    # print(cloud_info)
    # plt.figure()
    color = ['blue','red']
    pts_x,pts_y = [],[]
    for pt,h in zip(cloud_info,hit):
        if h==True:
            plt.scatter(pt[0],pt[1],c=color[1])
        else:
            plt.scatter(pt[0],pt[1],c=color[0],marker='v')
            pts_x.append(pt[0])
            pts_y.append(pt[1])
    # plt.plot(pts_x,pts_y)
    plt.savefig('/home/haochen/TPDM_transformer/wps/cloud_{}.png'.format(i))

def waypoint_printer(obs,i,ego):
    # waypoint = obs.road_waypoints
    ego_state = ego.ego_vehicle_state
    # neighborhood_vehicles = obs.neighborhood_vehicle_states
    plt.figure()
    # plt.scatter(ego_state.position[0], ego_state.position[1],marker='*')
    val = [ego]+list(obs.values())
    for l in range(6):
        n = val[l]

        # print(dir(n))
        # if n.ego_vehicle_state.id=='car_type_3-flow-route-edge-east-EN_1_random-edge-south-NS_0_random-8222204009439192561--7502074924325504014--9-2.0':
        #     print('ok')
        #     plt.scatter(n.ego_vehicle_state.position[0],n.ego_vehicle_state.position[1],s=100,marker='v',color='orange')
        waypoint = n.waypoint_paths
        color = 'orange' if l!=0 else 'red'
        plt.scatter(n.ego_vehicle_state.position[0],n.ego_vehicle_state.position[1],marker='*',color=color,s=50)
        cnt = 0
        # for k,wps in waypoint.lanes.items():
        wps = waypoint   
        # print(len(wps),n.ego_vehicle_state.id)
        for wp_list in wps:
            x,y = [],[]
            print(len(wp_list))
            for wp in wp_list:
                # print(wp)
                # assert 1==0
                x.append(wp.pos[0])
                y.append(wp.pos[1])
            # print(len(x))
            plt.scatter(x[0], y[0],color='black',s=5)
            plt.scatter(x[1:-1], y[1:-1],color='blue',s=5)
            plt.scatter(x[-1], y[-1],color='red',s=10)
            plt.plot(x,y,alpha=0.6)
            cnt += len(x)
            # break
        # print(cnt)
    plt.savefig('/home/haochen/TPDM_transformer/wps/wp_{}.png'.format(i))

def wp_processor(waypoint,ego_pos):
    pass

def path_printer(paths):
    plt.figure()
    for path in paths:
        x,y = [],[]
        for wp in path:
            x.append(wp.pos[0])
            y.append(wp.pos[1])
        plt.scatter(x, y)
        plt.plot(x,y)
    plt.savefig('/home/haochen/TPDM_transformer/path.png')

# get_state(obs['Agent-LHC'])
test_wps()
# test_str()