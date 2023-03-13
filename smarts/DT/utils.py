import numpy as np
import pandas as pd

from envision.client import Client as Envision
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns

def decode_map_xml(path,fig_size=(10,10),grid=False):
    # network = SumoRoadNetwork.from_file(path)
    network = SumoRoadNetwork(graph, net_file, map_spec)
    polys = network._compute_road_polygons()
    f,ax = plt.subplots(figsize=fig_size)


    cnt = 0
    for i,poly in enumerate(polys):
        p = poly.exterior.coords
        x,y = [c[0] for c in p],[c[1] for c in p]
        # h_alpha =  1
        # h_lw = 1.5
        plt.plot(x,y,'--', color='k', linewidth=1.5, alpha=1)
        cnt+=len(x)
    return plt

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape,(x.shape,self._M.shape)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True, clip=None,gamma=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.shape = shape
        self.rs = RunningStat(self.shape)
        self.gamma=gamma
        if gamma:
            self.ret = np.zeros(shape)

        
        # self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        # print(x)
        if self.gamma:
            self.ret = self.ret * self.gamma + x
            self.rs.push(self.ret)
        else:
            self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
                # x = x/(self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        # self.prev_filter.reset()
        if self.gamma:
            self.ret = np.zeros_like(self.ret)
        self.rs = RunningStat(self.shape)

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass
    
def test_r():
    r_filter = ZFilter(shape=(), center=False)

    r_list = []
    for r in range(10):
        rew = r_filter(r)
        print(r,rew)
    
    r_filter.reset()
    for r in range(10):
        rew = r_filter(r)
        print(r,rew)

class NeighbourAgentBuffer(object):
    def __init__(self,state_shape,hist_length=5,future_length=5,query_mode='full_future'):
        self.hist_length = hist_length
        self.future_length = future_length

        self.query_mode = query_mode
        assert self.query_mode in {'full_future','default','history_only'}
        #full future(the neighbors must have full future length steps given current timesteps)
        #TODO:default(the neighbors must have at least 1 future step)(involves using mask in calculating loss)
        self.buffer = dict()
        self.state_shape = state_shape

    
    def add(self,ids,values,timesteps):

        if ids not in self.buffer:
            self.buffer[ids]={
                'values':[],
                'timesteps':[]
            }

        # for easier query
        if len(self.buffer[ids]['timesteps'])>0:
            if timesteps!=self.buffer[ids]['timesteps'][-1]+1:
                # print(self.buffer[ids]['timesteps'][-1],timesteps)
                id_arr = np.arange(self.buffer[ids]['timesteps'][-1],timesteps+1)
                fp = np.array([self.buffer[ids]['values'][-1],values])

                res = []
                for i in range(fp.shape[1]):
                    inter = np.interp(id_arr[1:-1],[id_arr[0],id_arr[-1]],fp[:,i])
                    res.append(inter)
                res = np.transpose(res)
                
                for t,v in zip(id_arr[1:-1],res):
                    self.buffer[ids]['values'].append(v)
                    self.buffer[ids]['timesteps'].append(t)
                
                # print(self.buffer[ids]['timesteps'][-1])

            assert timesteps==self.buffer[ids]['timesteps'][-1]+1,('this_time steps:',timesteps,'last:',self.buffer[ids]['timesteps'][-1],ids)
        self.buffer[ids]['values'].append(values)
        self.buffer[ids]['timesteps'].append(timesteps)
    
    def query_neighbours(self,curr_timestep,curr_ids,curr_ind,keep_top=5,pad_length=10):
        neighbor_val = []
        i=0
        buf_ind=[]
        for ids,ind in zip(curr_ids,curr_ind):
            candidate_neighbor = self.buffer[ids]
            hist_t,fut_t = curr_timestep - candidate_neighbor['timesteps'][0] + 1,candidate_neighbor['timesteps'][-1]-curr_timestep
            n = max(hist_t-self.hist_length,0)
            l = min(hist_t,self.hist_length)
            f = min(fut_t,self.future_length)

            if self.query_mode=='history_only':
                if hist_t<=0:
                    continue
                val = candidate_neighbor['values'][n:n+l]
                # if l==1:
                #     val = np.expand_dims(val, axis=0)
                # print(np.array(val).shape)
                val = self.pad_hist(val,pad_length)
                # print(len(val),n,l)
                neighbor_val.append(val)
                buf_ind.append(ind)
                i+=1
            elif self.query_mode=='full_future':
                if fut_t<self.future_length:
                    continue
                hist = candidate_neighbor['values'][n:n+l]
                fut = candidate_neighbor['values'][n+l:n+l+f]
                hist = self.pad_hist(hist,pad_length)
                neighbor_val.append(hist+fut)
                i+=1    
            else:
                raise NotImplementedError()
            
            if i>=keep_top:
                break
        
        pad_num = keep_top - min(len(neighbor_val),keep_top)


        pad_val = np.zeros((np.clip(curr_timestep+1,0,self.state_shape[1]),self.state_shape[2]))
        # print(pad_val.shape,self.state_shape,curr_timestep)
        # print(neighbor_val[0].shape)

        neighbor = neighbor_val + pad_num*[pad_val]

        # print(neighbor)

        return neighbor,buf_ind


    def pad_hist(self,line,pad_length):
        assert len(np.array(line).shape)==2,(line)
        num = pad_length - min(pad_length,len(line))
        padded = [[0]*len(line[0])]*num + line
        
        return padded
    
    def clear(self):
        self.buffer = dict()

def split_future(egos,future_steps=10):
    res= []
    masks = []
    for i in range(egos.shape[0]):
        line = egos[i:i+future_steps]
        mask = [1]*line.shape[0] + [0]*(future_steps-line.shape[0])
        if line.shape[0]<future_steps:
            zeros = np.zeros((future_steps-line.shape[0],line.shape[-1]))
            line = np.concatenate((line,zeros),axis=0)
        
        res.append(line)
        masks.append(mask)
    
    return np.array(res) , np.array(masks)

def test_df():
    data = [8,9,10,11,12,13,14,15,16]
    t = 10
    hist,future = t-data[0]+1,data[-1]-t
    n = max(hist-10,0)
    l = min(hist,10)
    f = min(future,30)
    print(data[n:n+l])
    print(data[n+l:n+l+f])


if __name__=="__main__":
    test_df()
