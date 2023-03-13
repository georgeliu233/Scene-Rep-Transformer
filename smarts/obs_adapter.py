from collections import deque
from utils import NeighbourAgentBuffer
import numpy as np
import gym
import math
from copy import copy

class ObsAdapter(object):
    def __init__(self,mode,N_steps,dim,neighbors,query_mode='history_only',
        prediction=False,return_map=False):

        self.prediction=prediction
        self.return_map=return_map

        self.states = deque(maxlen=N_steps)
        self.epi_steps = 0
        self.test_states = deque(maxlen=N_steps)
        self.test_epi_steps = 0

        self.reward_steps = 0

        self.imgs = deque(maxlen=N_steps)
        self.test_imgs = deque(maxlen=N_steps)

        self.mode = mode
        self.N_steps = N_steps
        self.neighbors = neighbors

        if self.mode=='CNN' or self.mode=='CNN_LSTM':
            if mode=='CNN':
                OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(80, 80, 3*N_steps))
            else:
                OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(N_steps,80, 80, 3))
        
        elif mode=='STATE_LSTM' or mode=='Ego_surr':
            if mode=='STATE_LSTM':
                OBSERVATION_SPACE= gym.spaces.Box(low=-1000, high=1000, shape=(N_steps,dim*(neighbors+1)))
            elif mode=='STATE':
                OBSERVATION_SPACE=gym.spaces.Box(low=-1000, high=1000, shape=(N_steps*(neighbors+1)*dim,))
            else:
                OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(neighbors+1,N_steps,dim))
        elif mode=='STATE':
            OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(16,))
        else:
            raise NotImplementedError()
        
        self.OBSERVATION_SPACE = OBSERVATION_SPACE

        self.neighbours_buffer = NeighbourAgentBuffer(state_shape=(neighbors+1,N_steps,dim),hist_length=N_steps,future_length=10,query_mode=query_mode)
        self.test_neighbours_buffer = NeighbourAgentBuffer(state_shape=(neighbors+1,N_steps,dim),hist_length=N_steps,future_length=10,query_mode=query_mode)
    
    def get_obs_adapter(self):
        if self.mode=='CNN' or self.mode=='CNN_LSTM':
            return self.OBSERVATION_SPACE , self.observation_adapter , self.test_observation_adapter
        elif self.mode=='STATE_LSTM' or self.mode=='Ego_surr':
            return self.OBSERVATION_SPACE , self.neighbor_adapter , self.test_neighbor_adapter
        elif self.mode=='STATE':
            return self.OBSERVATION_SPACE , self.pure_state_adapter , self.pure_state_adapter
        else:
            raise NotImplementedError()
    
    
    def pure_state_adapter(self,env_obs):
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        # distance of vehicle from center of lane
        # closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        dist_from_centers = []
        angle_errors = []
        if len(wps)<3:
            for _ in range(3-len(wps)):
                dist_from_centers.append(-1)
                angle_errors.append(-1)
        for wp in wps:
            signed_dist_from_center = wp.signed_lateral_error(ego.position)
            lane_hwidth = wp.lane_width * 0.5
            dist_from_centers.append(signed_dist_from_center / lane_hwidth)
            angle_errors.append(wp.relative_heading(ego.heading))

        neighborhood_vehicles = env_obs.neighborhood_vehicle_states
        relative_neighbor_distance = [np.array([10, 10])]*3

        # no neighborhood vechicle
        if neighborhood_vehicles == None or len(neighborhood_vehicles) == 0:
            relative_neighbor_distance = [
                distance.tolist() for distance in relative_neighbor_distance]
        else:
            position_differences = np.array([math.pow(ego.position[0]-neighborhood_vehicle.position[0], 2) +
                                            math.pow(ego.position[1]-neighborhood_vehicle.position[1], 2) for neighborhood_vehicle in neighborhood_vehicles])

            nearest_vehicle_indexes = np.argsort(position_differences)
            for i in range(min(3, nearest_vehicle_indexes.shape[0])):
                relative_neighbor_distance[i] = np.clip(
                    (ego.position[:2]-neighborhood_vehicles[nearest_vehicle_indexes[i]].position[:2]), -10, 10).tolist()

        distances = [
                diff for diffs in relative_neighbor_distance for diff in diffs]
        observations =  np.array(
            dist_from_centers + angle_errors+ego.position[:2].tolist()+[ego.speed,ego.steering]+distances,
            dtype=np.float32,
        )
        assert observations.shape[-1]==16,observations.shape
        return observations

    def observation_adapter(self,env_obs):
        # print(env_obs)
        new_obs = env_obs.top_down_rgb[1] / 255.0
        
        self.states.append(new_obs)
        res = np.array(list(self.states))

        if res.shape[0]<self.N_steps:
            res = np.concatenate((res,np.zeros(shape=(self.N_steps-res.shape[0],)+res.shape[1:])),axis=0)
        
        #[timesteps,80,80,3]
        if self.mode=='CNN':
            #[80,80,3*timesteps]
            res = np.transpose(res,[1,2,3,0]).reshape(self.OBSERVATION_SPACE.shape)
        self.epi_steps +=1

        if  (env_obs.events.collisions 
            or env_obs.events.reached_goal 
            or env_obs.events.agents_alive_done 
            or env_obs.events.reached_max_episode_steps
            or env_obs.events.off_road):
            
            self.states=deque(maxlen=self.N_steps)
            self.epi_steps = 0
        
        if self.prediction:
            ego = env_obs.ego_vehicle_state
            ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
            return np.array(res, dtype=np.float32),np.array(ego_state, dtype=np.float32)
        return np.array(res, dtype=np.float32)
    
    def test_observation_adapter(self,env_obs):

        new_obs = env_obs.top_down_rgb[1] / 255.0
        
        self.test_states.append(new_obs)
        res = np.array(list(self.test_states))

        if res.shape[0]<self.N_steps:
            res = np.concatenate((res,np.zeros(shape=(self.N_steps-res.shape[0],)+res.shape[1:])),axis=0)
        
        #[timesteps,80,80,3]
        if self.mode=='CNN':
            #[80,80,3*timesteps]
            res = np.transpose(res,[1,2,3,0]).reshape(self.OBSERVATION_SPACE.shape)
        self.test_epi_steps +=1

        if  (env_obs.events.collisions 
            or env_obs.events.reached_goal 
            or env_obs.events.agents_alive_done 
            or env_obs.events.reached_max_episode_steps
            or env_obs.events.off_road):
            
            self.test_states=deque(maxlen=self.N_steps)
            self.test_epi_steps = 0
        
        if self.prediction:
            ego = env_obs.ego_vehicle_state
            ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
            return np.array(res, dtype=np.float32),np.array(ego_state, dtype=np.float32)
            
        return np.array(res, dtype=np.float32)

    def neighbor_adapter(self,obs):

        ego= obs.ego_vehicle_state
        neighbours = obs.neighborhood_vehicle_states
        
        #x,y,psi
        ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
        
        self.states.append(ego_state)
        
        dis_list = []
        min_ind = []
        id_list = []
        # print(epi_steps)
        if len(neighbours)>0:  
            for neighbour in neighbours:
                x,y= neighbour.position[0],neighbour.position[1]
                psi = float(neighbour.heading)
                speed = neighbour.speed
                # print(x,y,ego_state[0],ego_state[1],dis)
                dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
                self.neighbours_buffer.add(ids=neighbour.id, values=[x,y,speed,dis,psi],timesteps=self.epi_steps)
                dis_list.append(dis)
                id_list.append(neighbour.id)
            
            min_ind = np.argsort(dis_list)

            n_id_list = [id_list[i] for i in min_ind]
            id_list = n_id_list

        if len(id_list)>0:
            neighbors_state = self.neighbours_buffer.query_neighbours(curr_timestep=self.epi_steps,curr_ids=id_list
            ,curr_ind=min_ind,keep_top=self.neighbors,pad_length=min(self.epi_steps+1,self.N_steps))
        else:
            neighbors_state = np.zeros((self.neighbors,self.OBSERVATION_SPACE.shape[1],self.OBSERVATION_SPACE.shape[2]))
        
        # if len(np.array(neighbors_state).shape)==2:
        #     print(neighbors_state)

        new_states =np.concatenate((np.expand_dims(list(self.states),0),neighbors_state),axis=0)
        
        if new_states.shape[1]<self.N_steps:
            padded = np.zeros((new_states.shape[0],self.N_steps-new_states.shape[1],new_states.shape[2]))
            new_states = np.concatenate((new_states,padded),axis=1)

        self.epi_steps +=1   

        if  (obs.events.collisions 
            or obs.events.reached_goal 
            or obs.events.agents_alive_done 
            or obs.events.reached_max_episode_steps
            or obs.events.off_road):

            self.states=deque(maxlen=self.N_steps)
            self.epi_steps = 0
            self.neighbours_buffer.clear()
        
        if self.mode=='STATE_LSTM':
            out = np.array(new_states, dtype=np.float32)
            out = np.transpose(out,(1,0,2))
            return np.reshape(out, (self.N_steps,-1))
        elif self.mode=='STATE':
            out = np.array(new_states, dtype=np.float32)
            out = np.transpose(out,(1,0,2))
            return np.reshape(out, (-1))

        return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32)

    def test_neighbor_adapter(self,obs):

        ego= obs.ego_vehicle_state
        neighbours = obs.neighborhood_vehicle_states
        
        #x,y,psi
        ego_state = [ego.position[0],ego.position[1],ego.speed,0.0,float(ego.heading)]
        
        self.test_states.append(ego_state)
        # print(ego_state)
        # print(len(neighbours),test_epi_steps)
        dis_list = []
        min_ind = []
        id_list = []
        # print(epi_steps)
        if len(neighbours)>0:  
            for neighbour in neighbours:
                x,y= neighbour.position[0],neighbour.position[1]
                psi = float(neighbour.heading)
                speed = neighbour.speed
                # print(x,y,ego_state[0],ego_state[1],dis)
                dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
                self.test_neighbours_buffer.add(ids=neighbour.id, values=[x,y,speed,dis,psi],timesteps=self.test_epi_steps)
                dis_list.append(dis)
                id_list.append(neighbour.id)
            
            min_ind = np.argsort(dis_list)

            n_id_list = [id_list[i] for i in min_ind]
            id_list = n_id_list

        if len(id_list)>0:
            neighbors_state = self.test_neighbours_buffer.query_neighbours(curr_timestep=self.test_epi_steps,curr_ids=id_list
            ,keep_top=self.neighbors,pad_length=min(self.test_epi_steps+1,self.N_steps))
        else:
            neighbors_state = np.zeros((self.neighbors,)+self.OBSERVATION_SPACE.shape[1:])

        new_states =np.concatenate((np.expand_dims(list(self.test_states),0),neighbors_state),axis=0)
        
        if new_states.shape[1]<self.N_steps:
            padded = np.zeros((new_states.shape[0],self.N_steps-new_states.shape[1],new_states.shape[2]))
            new_states = np.concatenate((new_states,padded),axis=1)

        self.test_epi_steps +=1   

        if  (obs.events.collisions 
            or obs.events.reached_goal 
            or obs.events.agents_alive_done 
            or obs.events.reached_max_episode_steps
            or obs.events.off_road):

            self.test_states=deque(maxlen=self.N_steps)
            self.test_epi_steps = 0
            self.test_neighbours_buffer.clear()
        
        if self.mode=='STATE_LSTM':
            out = np.array(new_states, dtype=np.float32)
            out = np.transpose(out,(1,0,2))
            return np.reshape(out, (self.N_steps,-1))
        elif self.mode=='STATE':
            out = np.array(new_states, dtype=np.float32)
            out = np.transpose(out,(1,0,2))
            return np.reshape(out, (-1))
        
        return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32)

class NeighbourObsAdapter(object):
    def __init__(self,mode,N_steps,dim,neighbors,query_mode='history_only',
        prediction=False,return_map=False,planned_path=10):

        self.prediction=prediction
        self.return_map=return_map

        self.states = deque(maxlen=N_steps)
        self.epi_steps = 0
        self.test_states = deque(maxlen=N_steps)
        self.test_epi_steps = 0

        self.reward_steps = 0

        self.imgs = deque(maxlen=N_steps)
        self.test_imgs = deque(maxlen=N_steps)

        self.mode = mode
        self.N_steps = N_steps
        self.neighbors = neighbors
        self.planned_path = planned_path

        self.OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(neighbors+1,N_steps,dim))

        self.neighbours_buffer = NeighbourAgentBuffer(state_shape=(neighbors+1,N_steps,dim),hist_length=N_steps,future_length=10,query_mode=query_mode)
        self.test_neighbours_buffer = NeighbourAgentBuffer(state_shape=(neighbors+1,N_steps,dim),hist_length=N_steps,future_length=10,query_mode=query_mode)

    def waypoint_adapter(self,wp_list,is_ego):
        res = []
        for wp in wp_list[:2]:
            line=[]
            for p in wp[1:1+self.planned_path]:
                x,y,heading = p.pos[0],p.pos[1],float(p.heading)
                if is_ego: 
                    a,b = 1.0,0.0
                else:
                    a,b = 0.0,1.0
                line.append([x,y,heading,a,b])
            # if len(line)!=self.planned_path:
            #     # if len(line)==0:
            #     #     last = [0,0,0,0,0]
            #     # else:
            #     #     last = [x,y,heading,a,b]
            # else:
            #     last = [0,0,0,0,0]
            last = [0,0,0,0,0]
            line = line+[last]*(self.planned_path-len(line))
            res.append(line)

        if len(res)==1:
            res.append(line)
        return np.array(res,np.float32)
    
    def simple_obs_adapter(self,obs,neighbour_obs,neighbor=5,max_distance=10):

        ego = obs.ego_vehicle_state
        ego_state = [ego.position[0],ego.position[1],float(ego.heading),ego.speed*np.cos(float(ego.heading)),ego.speed*np.sin(float(ego.heading))]
        relative_neighbor_distance = np.array([[max_distance,max_distance,0,0,0]]*neighbor)
        neighbour_obs = list(neighbour_obs.values())
        neighborhood_vehicles = neighbour_obs

        map_list = [self.waypoint_adapter(wp_list=obs.waypoint_paths, is_ego=True)]
        id_list = []

        if not(neighborhood_vehicles == None or len(neighborhood_vehicles) == 0):
            position_differences = np.array([math.pow(ego.position[0]-neighborhood_vehicle.ego_vehicle_state.position[0], 2) +
                    math.pow(ego.position[1]-neighborhood_vehicle.ego_vehicle_state.position[1], 2) for neighborhood_vehicle in neighborhood_vehicles])
            nearest_vehicle_indexes = np.argsort(position_differences)[:min(neighbor, position_differences.shape[0])]
            for i,ind in enumerate(nearest_vehicle_indexes):
                relative_neighbor_distance[i] = np.array(
                    [
                        np.clip(ego_state[0]-neighborhood_vehicles[ind].ego_vehicle_state.position[0],-max_distance,max_distance),
                        np.clip(ego_state[1]-neighborhood_vehicles[ind].ego_vehicle_state.position[1],-max_distance,max_distance),
                        ego_state[2]-float(neighborhood_vehicles[ind].ego_vehicle_state.heading),
                        neighborhood_vehicles[ind].ego_vehicle_state.speed*np.cos(float(neighborhood_vehicles[ind].ego_vehicle_state.heading)),
                        neighborhood_vehicles[ind].ego_vehicle_state.speed*np.sin(float(neighborhood_vehicles[ind].ego_vehicle_state.heading))
                    ]
                )
            id_list = nearest_vehicle_indexes
        # relative_neighbor_distance = np.reshape(relative_neighbor_distance, [-1])
        observation = np.concatenate((np.expand_dims(ego_state,0),relative_neighbor_distance))
        self.states.append(observation)
        for ids in id_list:
            wp_list = neighbour_obs[ids].waypoint_paths
            map_list.append(self.waypoint_adapter(wp_list, is_ego=False))
        if len(map_list)>=self.neighbors+1:
            map_state = np.concatenate(map_list,axis=0)
        else:
            map_state = np.concatenate(map_list+[np.zeros_like(map_list[0])]*(self.neighbors-(len(map_list)-1)),axis=0)

        new_states =np.array(list(self.states)).transpose(1,0,2)
        
        if new_states.shape[1]<self.N_steps:
            padded = np.zeros((new_states.shape[0],self.N_steps-new_states.shape[1],new_states.shape[2]))
            new_states = np.concatenate((new_states,padded),axis=1)

        self.epi_steps +=1   

        if  (obs.events.collisions 
            or obs.events.reached_goal 
            or obs.events.agents_alive_done 
            or obs.events.reached_max_episode_steps
            or obs.events.off_road):

            self.states=deque(maxlen=self.N_steps)
            self.epi_steps = 0
            self.neighbours_buffer.clear()

        return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32),map_state

    def obs_adapter(self,obs,neighbour_obs):

        ego= obs.ego_vehicle_state
        ego_speed,ego_psi = ego.speed,float(ego.heading)
        ego_state = [ego.position[0],ego.position[1],ego_psi,ego_speed*np.cos(ego_psi),ego_speed*np.sin(ego_psi)]

        self.states.append(ego_state)
        
        dis_list = []
        min_ind = []
        id_list = []
        #(neighbor+1,2,10,2)
        map_list = [self.waypoint_adapter(wp_list=obs.waypoint_paths, is_ego=True)]
        # print(epi_steps)
        neighbour_obs = list(neighbour_obs.values())
        if len(neighbour_obs)>0:  
            for neighbour in neighbour_obs:
                x,y= neighbour.ego_vehicle_state.position[0],neighbour.ego_vehicle_state.position[1]
                psi = float(neighbour.ego_vehicle_state.heading)
                speed = neighbour.ego_vehicle_state.speed
                dis = np.sqrt((ego_state[0]-x)**2 + (ego_state[1]-y)**2)
                self.neighbours_buffer.add(ids=neighbour.ego_vehicle_state.id, values=[x,y,psi,speed*np.cos(psi),speed*np.sin(psi)],timesteps=self.epi_steps)
                # if dis<=10:
                dis_list.append(dis)
                id_list.append(neighbour.ego_vehicle_state.id)
                # map_list.append(self.waypoint_adapter(wp_list=neighbour.waypoint_paths, is_ego=False))
            
            min_ind = np.argsort(dis_list)

            n_id_list = [id_list[i] for i in min_ind]
            id_list = n_id_list

        if len(id_list)>0:
            neighbors_state,buf_ind = self.neighbours_buffer.query_neighbours(curr_timestep=self.epi_steps,curr_ids=id_list,
            curr_ind=min_ind,keep_top=self.neighbors,pad_length=min(self.epi_steps+1,self.N_steps))
        else:
            neighbors_state = np.zeros((self.neighbors,len(list(self.states)),self.OBSERVATION_SPACE.shape[2]))
            buf_ind=[]

        for ids in buf_ind:
            wp_list = neighbour_obs[ids].waypoint_paths
            map_list.append(self.waypoint_adapter(wp_list, is_ego=False))
        if len(map_list)>=self.neighbors+1:
            map_state = np.concatenate(map_list,axis=0)
        else:
            map_state = np.concatenate(map_list+[np.zeros_like(map_list[0])]*(self.neighbors-(len(map_list)-1)),axis=0)
        

        new_states =np.concatenate((np.expand_dims(list(self.states),0),neighbors_state),axis=0)
        
        if new_states.shape[1]<self.N_steps:
            padded = np.zeros((new_states.shape[0],self.N_steps-new_states.shape[1],new_states.shape[2]))
            new_states = np.concatenate((new_states,padded),axis=1)

        self.epi_steps +=1   

        if  (obs.events.collisions 
            or obs.events.reached_goal 
            or obs.events.agents_alive_done 
            or obs.events.reached_max_episode_steps
            or obs.events.off_road):

            self.states=deque(maxlen=self.N_steps)
            self.epi_steps = 0
            self.neighbours_buffer.clear()
        
        # if self.mode=='STATE_LSTM':
        #     out = np.array(new_states, dtype=np.float32)
        #     out = np.transpose(out,(1,0,2))
        #     return np.reshape(out, (self.N_steps,-1))
        # elif self.mode=='STATE':
        #     out = np.array(new_states, dtype=np.float32)
        #     out = np.transpose(out,(1,0,2))
        #     return np.reshape(out, (-1))

        return np.array(new_states, dtype=np.float32),np.array(ego_state, dtype=np.float32),map_state
    


