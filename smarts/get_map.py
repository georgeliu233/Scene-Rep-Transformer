from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle


def decode_map_xml(path):
    network = SumoRoadNetwork.from_file(path)
    graph = network.graph
    lanepoints = network._lanepoints
    nodes = graph.getNodes()
    # print(nodes)
    # print(graph.getEdges())
    polys = []
    print(len(graph.getEdges()))
    for edge in graph.getEdges():
        poly = []
        print(len(edge.getLanes()))
        for lane in edge.getLanes():
            # shape = SumoRoadNetwork._buffered_lane_or_edge(lane, lane.getWidth())
            shape = lane.getShape()
            print(lane.getID())
            # print(lane.getParams())
            # print(lane.getNeigh())
            print(lane.getWidth())
            print(lane.getBoundingBox())
            # print(lane.getOutgoing())
            # print(lane.getIncoming())
            # print(lane.getConnection())
            poly.append(shape)
            # Check if "shape" is just a point.
            # if len(set(shape.exterior.coords)) == 1:
            #     # logging.debug(
            #     #     f"Lane:{lane.getID()} has provided non-shape values {lane.getShape()}"
            #     # )
            #     continue
        polys.append(poly)

    print(len(polys))
    cnt = 0
    for i,poly in enumerate(polys):
        for p in poly:
            x,y = [c[0] for c in p],[c[1] for c in p]
            x,y = make_interp(x, y)
            print(len(x))
            plt.scatter(x[0], y[0],edgecolors='black')
            plt.plot(x,y)
            plt.scatter(x,y,s=10)
            cnt+=len(x)
    print(cnt)

def make_interp(x_value,y_value,min_dist=2):
    interp_x = []
    interp_y = []
    for j in range(len(x_value)-1):
        x_diff = x_value[j+1] - x_value[j]
        y_diff = y_value[j+1] - y_value[j]
        dist=np.sqrt(x_diff**2+y_diff**2)
        if dist<=min_dist:
            interp_x.append(x_value[j])
            interp_y.append(y_value[j])
        else:
            need_interp_num = dist//min_dist
            index = np.arange(2+need_interp_num)
            new_x = np.interp(index,[0,index[-1]],[x_value[j],x_value[j+1]]).tolist()
            new_y = np.interp(index,[0,index[-1]],[y_value[j],y_value[j+1]]).tolist()
            interp_x = interp_x + new_x[:-1] #traj.x_value[j+1] doesnot count
            interp_y = interp_y + new_y[:-1]
        
    interp_x.append(x_value[-1])
    interp_y.append(y_value[-1])

    return interp_x,interp_y

def process_map(path):
    network = SumoRoadNetwork.from_file(path)
    graph = network.graph
    lanepoints = network._lanepoints
    nodes = graph.getNodes()
    polys = []
    for edge in tqdm(graph.getEdges()):
        poly = []
        for lane in edge.getLanes():
            shape = lane.getShape()
            width = getWidth()
            ID = lane.getID()
            x,y = [s[0] for s in shape],[s[1] for s in shape]
            x,y = make_interp(x, y)
            poly.append([x,y,width,ID])
        polys.append(poly)
    

if __name__=='__main__':
    path = ''
    decode_map_xml(path)
