import pygame
import pygame.freetype

import weakref
import logging
import random
import collections
import numpy as np
import math
import cv2
import re
import sys
import os
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# append sys PATH for CARLA simulator 
sys.path.append('xxx/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
sys.path.append('xxx/CARLA_0.9.13/PythonAPI/carla/')

import carla
from carla import ColorConverter as cc
from agents.navigation.basic_agent import BasicAgent
from carla import VehicleLightState as vls

screen_width, screen_height = 640, 360 #long, short
WIDTH, HEIGHT, PACK = 80, 45, 4

class InterSection(object):
    def __init__(self, enabled_obs_number=8, vehicle_type = 'single', use_checker = False,
                 control_interval = 1, advanced_info = False,
                 surrounding_record = False, frame=10, port=2200, 
                 seed=0):

        self.image_size = WIDTH * HEIGHT
        self.action_size = 1

        ## set the carla World parameters
        self.vehicle_type = vehicle_type
        self.control_interval = control_interval
        self.advanced_info = advanced_info
        self.use_checker = use_checker
        self.surrounding_record = surrounding_record
        self.frame = frame

        ## set the actors
        self.ego_vehicle = None
        self.obs_list, self.bp_obs_list, self.spawn_point_obs_list = [], [] ,[]
        self.maximum_enabled_obs = 8
        self.enabled_obs = enabled_obs_number if enabled_obs_number<=self.maximum_enabled_obs else self.maximum_enabled_obs

        ## set the sensory actors
        self.collision_sensor = None
        self.seman_camera = None
        self.viz_camera = None
        self.surface = None
        self.camera_output = np.zeros([360,640,3])
        self.camera_output1 = np.zeros([360,640,3])
        self.recording = False
        self.Attachment = carla.AttachmentType

        ## connect to the CARLA client
        self.port = port
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(20.0)
        
        ## build the CARLA world
        self.world = self.client.load_world('Town10HD_Opt')
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.map = self.world.get_map()
        
        self._weather_presets = find_weather_presets()
        self._weather_index = 8

        ## initialize the pygame settings

        settings = self.world.get_settings()
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        
        self.seed = seed
        # self.reset()

    def reset(self):

        settings = self.world.get_settings()
        self.original_settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.frame
        self.world.apply_settings(settings)
        
        ## reset the recording lists
        self.steer_history = []
        self.intervene_history = []
        self.lat_action_history = []
        self.target_speed_history = []
        self.ego_location_history = deque(maxlen=10)
        self._steer_cache = 0
        self.ppp = None
        self.ppp1 = None
        self.y_aver = None
        self.dist_travelled = 0

        ## reset the human intervention state
        self.intervention = False
        self.risk = None
        self.v_upp = 19.5/7
        self.v_low = 13.5/7
        self.ii = None
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor
        
        self.traffic_manager = self.client.get_trafficmanager(self.port+50)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_random_device_seed(self.seed)
        self.seed = (self.seed + 1) % 500
        self.traffic_manager.set_synchronous_mode(True)
       
        synchronous_master = False
        
        list_actor = self.world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                actor_.set_state(carla.TrafficLightState.Green) 
                actor_.set_green_time(2000.0)
                
        ## spawn the ego vehicle (fixed)
        bp_ego = self.world.get_blueprint_library().filter('vehicle.mercedes.coupe_2020')[0]
        bp_ego.set_attribute('color', '0, 0, 0')
        bp_ego.set_attribute('role_name', 'hero')

        spawn_point_ego = self.world.get_map().get_spawn_points()[0]
        spawn_point_ego.location.x = 0
        spawn_point_ego.location.y = -64.5
        spawn_point_ego.location.z  = 0.1
        spawn_point_ego.rotation.yaw = 180

        if self.ego_vehicle is not None:
            self.destroy()
        self.ego_vehicle = self.world.spawn_actor(bp_ego, spawn_point_ego)
        
        self.ego_current_speed_ratio = -100
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, self.ego_current_speed_ratio)
        
        self.target_speed = 48
        
        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        self.world.tick()
        
        self.agent = BasicAgent(self.ego_vehicle, target_speed = self.target_speed)

        self.speed_limit_flag = 0
        l = self.ego_vehicle.bounding_box.extent.x*0.9
        w = self.ego_vehicle.bounding_box.extent.y
        self.fix_theta = np.arctan(w/l) * 180 / np.pi
        self.fix_length = np.sqrt(l**2+w**2)
        self.displacement_waypoint = self.map.get_waypoint(spawn_point_ego.location)
        self.waypoint_ego = self.map.get_waypoint(spawn_point_ego.location)

        self.count = 0
        self.subcount1 = 0
        self.subcount2 = 0
        self.interval = 2
        self.command_interval = round(self.frame / self.interval) # execute 2 command per second
        self.list_action = []

        ## Spawn surronding vehicles
        self.obs_list = []
        self.obs_velo_list = []
        self.obs_agent_list = []

        # randomly choose spawned points
        lat_list = [-71, -70,
                    -97, -102, 
                    -107, -103,
                    
                    -45.5,-45.5,
                    -23.5, -12]
        long_list = [-61.5, -57.5,
                     -42.5, -40.5, 
                     -2.5, -2,
                     
                     -24.5, -35,
                     -68.2, -68.2]
        yaw_list = [0.0, 0.0,
                    -61.5, -63.7, 
                    -90, -90,

                    -90,-90,
                    180,180]
        spawn_points = []
        
        random_vehicle_indexes = np.random.choice(len(lat_list), len(lat_list), replace=False)
        random_vehicle_indexes = sorted(random_vehicle_indexes)
        for i in random_vehicle_indexes:
            trans = carla.Transform()
            trans.location.x, trans.location.y, trans.location.z = lat_list[i], long_list[i], 0.1
            trans.rotation.yaw = yaw_list[i]
            spawn_points.append(trans)
        
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if (int(x.get_attribute('number_of_wheels')) == 4 and x.id != 'vehicle.volkswagen.t2' and x.id != 'vehicle.bmw.isetta')
                      and x.id != 'vehicle.carlamotors.*' and x.id=='vehicle.tesla.model3']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        # randomly spwan actors in the random chosen spawned points
        batch = []
        for n, transform in enumerate(spawn_points):
            bp_sv = random.choice(blueprints)
            if bp_sv.has_attribute('color'):
                color = random.choice(bp_sv.get_attribute('color').recommended_values)
                bp_sv.set_attribute('color', color)
            if bp_sv.has_attribute('driver_id'):
                driver_id = random.choice(bp_sv.get_attribute('driver_id').recommended_values)
                bp_sv.set_attribute('driver_id', driver_id)
            bp_sv.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            light_state = vls.RightBlinker | vls.LeftBlinker | vls.Brake
                
            batch.append(SpawnActor(bp_sv, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.obs_list.append(response.actor_id)
        
            
        ## Spawn walkers randomly
        self.walker_list = []
        self.walker_id = []
        walker_spawn_points = []
        self.surrounding_number_walker = 2
        x = [-60, -62]
        y = [-47.5, -48.5]
        
        for i in range(self.surrounding_number_walker):
            spawn_point = carla.Transform()
            spawn_point.location.x = x[i]
            spawn_point.location.y = y[i]
            spawn_point.location.z = 1
            spawn_point.rotation.yaw = 0
            walker_spawn_points.append(spawn_point)
        
        walker_batch = []
        self.walker_speed = []
        percentagePedestriansRunning = 0
        blueprints_walkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(blueprints_walkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    self.walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    self.walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                # print("Walker has no speed")
                self.walker_speed.append(0.0)
            walker_batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(walker_batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walker_list.append({"id": results[i].actor_id})
        
        self.walker_direction = []
        for i in range(len(self.walker_list)):
            td = carla.Vector3D()
            td.x = 1
            self.walker_direction.append(td)
            
        for i in range(len(self.walker_list)):
            self.walker_id.append(self.walker_list[i]["id"])
            
        self.walkers = self.world.get_actors(self.walker_id)
    
    
        ######### VERY IMPORTANT, METHOD FOUND BY JINGDA!!!!!!! ########
        self.obs_actors = self.world.get_actors(self.obs_list)

        iii = 0
        for v in self.obs_actors:
            self.traffic_manager.auto_lane_change(v,True)
            self.traffic_manager.vehicle_percentage_speed_difference(v,np.random.randint(-50,-20))
            self.traffic_manager.distance_to_leading_vehicle(v, np.random.randint(8,12))
            iii += 1
        
        self.speed_limit_obs_flags = np.zeros(iii)
        
        ## configurate and spawn the collision sensor
        # clear the collision history list
        self.collision_history = []
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        # spawn the collision sensor actor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        self.collision_sensor = self.world.spawn_actor(
                bp_collision, carla.Transform(), attach_to=self.ego_vehicle)
        # obtain the collision signal and append to the history list
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: InterSection._on_collision(weak_self, event))
        
        ## reset the step counter
        self.count = 0
        self.count_yaw = 0
        self.reset_traj_dataset()
        # interpolated waypoints for this scenario map, otherwise perform unrealistic lane-change
        script_dir = os.path.dirname(__file__)
        self.wp = np.load(script_dir+'/map/wp.npy')
        self.wp2 = np.load(script_dir+'/map/wp2.npy')
        
        state = self.get_observation_scene()
        return state

    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_history.append((event.frame, intensity))
        if len(self.collision_history) > 4000:
            self.collision_history.pop(0)

    def get_collision_history(self):
        collision_history = collections.defaultdict(int)
        flag = 0
        for frame, intensity in self.collision_history:
            collision_history[frame] += intensity
            if intensity != 0:
                flag = 1
        return collision_history, flag
    
    ## search waypoints
    def get_position(self, waypoint):
        loc = waypoint.transform.location
        return [(loc.x, loc.y)]
    
    def depth_first_search(self, curr_waypoint, depth=0, max_depth=49):
        if depth > max_depth:
            return [self.get_position(curr_waypoint)]
        else:
            trasversed_lanes = []
            child_lanes = curr_waypoint.next(0.5)
            if len(child_lanes) > 0:
                for child in child_lanes:
                    trajs = self.depth_first_search(child, depth+1, max_depth)
                    trasversed_lanes.extend(trajs)
            if len(trasversed_lanes) == 0:
                return [self.get_position(curr_waypoint)]
            
            res = []
            for lane in trasversed_lanes:
                res.append(self.get_position(curr_waypoint) + lane)
            return res 
    
    def filter_and_pad(self, all_results, vehicle_location, k=3, length=50):
        lane_position = {}
        for i, result in enumerate(all_results):
            lane_position[i] = np.min(np.linalg.norm(np.array(result)-np.array(vehicle_location)[np.newaxis,:]) )
        sort_lanes = sorted(lane_position.items(), key=lambda x:x[1])[:k]
        
        new_result = np.zeros((k, length, 2))
        for i, lane in enumerate(sort_lanes):
            select_lane = np.array(all_results[lane[0]])[:length]
            new_result[i] = np.pad(select_lane, pad_width=[[0, length-select_lane.shape[0]], [0, 0]])
        return new_result
    
    def fitler_goal_waypoints(self,results, goal, preview_dis):
        goal = np.array(goal)[np.newaxis,:]
        min_dist = []
        for result in results:
            m_dist = np.min(np.linalg.norm(np.array(result)-goal,axis=-1))
            min_dist.append(m_dist)
        arg = np.argmin(np.array(min_dist))
        traj = np.array(results[arg])
        return results[arg][preview_dis]
    
    def filter_initial_waypoints(self,result, ego_location, preview_dis):
        ego_location = np.array(ego_location)[np.newaxis,:]
        m_dist = np.argmin(np.linalg.norm(np.array(result)-ego_location,axis=-1))
        self.ego_wp = self.filter_and_pad([result], ego_location)
        return result[m_dist + preview_dis]
    

    def filter_planned_ego_waypoints(self, vehicle, preview_dis):
        location = vehicle.get_location()
        ego_location = [location.x, location.y]
        ego_location = np.array(ego_location)[np.newaxis,:]

        dist_1 = np.linalg.norm(self.wp-ego_location,axis=-1)
        dist_2 = np.linalg.norm(self.wp2-ego_location,axis=-1)
        min_d1, arg_md1 = np.min(dist_1), np.argmin(dist_1)
        min_d2, arg_md2 = np.min(dist_2), np.argmin(dist_2)
        if min_d1 < min_d2:
            r = self.wp[arg_md1 + preview_dis]
            rr = self.wp2[arg_md2 + preview_dis + 2]
            lr = None
        else:
            r = self.wp2[arg_md2 + preview_dis]
            lr = self.wp[arg_md1 + preview_dis + 2]
            rr = None
        
        return lr, r, rr

    
    def filter_ego_waypoints(self,vehicle, preview_dis):
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        vehicle_location = [location.x, location.y]
        left_results, right_results = None, None
        
        goal = [-52.5, -32]
        results = self.depth_first_search(waypoint,max_depth=200)

        # plt.scatter(goal[0],goal[1],s=50, color='r')
        if vehicle_location[1]<-50:
            # r = self.fitler_goal_waypoints(results, goal, preview_dis)
            r = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)
        else:
            r = self.fitler_goal_waypoints(results, goal, preview_dis)
        # plt.scatter(r[0],r[1],s=20, color='b')
        lr, rr = None, None
        
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None):
            
            if vehicle_location[1]<-50:
                lr = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)
            else:
                left_results = self.depth_first_search(waypoint.get_left_lane(),max_depth=200)
                lr = self.fitler_goal_waypoints(left_results, goal, preview_dis)
            # plt.scatter(lr[0],lr[1],s=20, color='b')
      
        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None):
            if vehicle_location[1]<-50:
                rr = self.filter_initial_waypoints(self.wp,vehicle_location,preview_dis)
            else:
                right_results = self.depth_first_search(waypoint.get_right_lane(),max_depth=200)
                rr = self.fitler_goal_waypoints(right_results, goal, preview_dis)

        return lr, r, rr

    
    def get_all_waypoints(self, vehicle,judge=False):
        location = vehicle.get_location()
        waypoint = self.map.get_waypoint(location)
        vehicle_location = [location.x, location.y]
        
        left_results, right_results = None, None
        results = self.depth_first_search(waypoint)
        if judge:
            self.judge_off_route(location.x, location.y, results)
        if (waypoint.lane_change & carla.LaneChange.Left != 0) and (waypoint.get_left_lane() is not None):
            left_results = self.depth_first_search(waypoint.get_left_lane())
            results.extend(left_results)
        if (waypoint.lane_change & carla.LaneChange.Right != 0) and (waypoint.get_right_lane() is not None):
            right_results = self.depth_first_search(waypoint.get_right_lane())
            results.extend(right_results)
        
        goal = [-52.5, -32]
        new_results = self.filter_and_pad(results, goal)
        return new_results
    
    def get_walker_waypoint(self, walker):
        x_walker, y_walker = walker.get_location().x, walker.get_location().y
        x = np.arange(0, 25.0, 0.5) + x_walker
        y = [y_walker]*50
        traj = np.stack([x, y],axis=1)
        return traj
    
    def select_top_actors(self, actors, walkers, vehicle_location, k=5):
        lane_position = {}
        for i, act in enumerate(actors):
            act_position = act.get_location()
            pos = [act_position.x, act_position.y]
            lane_position[i] = [np.linalg.norm(pos-np.array(vehicle_location)), 0] 
        for i, act in enumerate(walkers):
            act_position = act.get_location()
            pos = [act_position.x, act_position.y]
            lane_position[i] = [np.linalg.norm(pos-np.array(vehicle_location)), 1] 
        sort_lanes = sorted(lane_position.items(), key=lambda x:x[1][0])[:k]
        return sort_lanes
    
    def reset_traj_dataset(self):
        self.traj_dataset = defaultdict()
        self.traj_dataset['ego'] = dict()
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_'+str(obs_id)] = dict()
        for obs_id in range(len(self.walkers)):
            self.traj_dataset['w_'+str(obs_id)] = dict()
    
    
    def angle_norm(self, yaw):
        theta = yaw - 90
        return (theta*np.pi/180 + np.pi) % (2*np.pi) - np.pi

    
    def get_actor_state(self, actor, types):
        return [actor.get_location().x, actor.get_location().y,
                     self.angle_norm(actor.get_transform().rotation.yaw),
                     actor.get_velocity().x, actor.get_velocity().y]
    
    def record_one_step(self):
        self.traj_dataset['ego'][self.count] = self.get_actor_state(self.ego_vehicle,0)
        for obs_id in range(len(self.obs_actors)):
            self.traj_dataset['v_'+str(obs_id)][self.count] = self.get_actor_state(self.obs_actors[obs_id],1)
        for obs_id in range(len(self.walkers)):
            self.traj_dataset['w_'+str(obs_id)][self.count] = self.get_actor_state(self.walkers[obs_id], 2)
    
    def query_single_trajs(self,name):
        self_trajs = np.zeros((10, 5))
        queryed_trajs = self.traj_dataset[name]
        for i in range(10):
            queryed_time = self.count - i
            if queryed_time in queryed_trajs:
                self_trajs[-i, :] = np.array(queryed_trajs[queryed_time])
        return self_trajs
    
    def get_observation_scene(self):
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        self.record_one_step()
        self.ego_location_history.append([x_ego, y_ego])
        if len(self.ego_location_history)==1:
            step_dist = 0
        else:
            step_dist = np.sqrt((self.ego_location_history[-2][0]-x_ego)**2 +(self.ego_location_history[-2][1]-y_ego)**2 )
        self.dist_travelled += step_dist
        
        ego_waypoint = self.get_all_waypoints(self.ego_vehicle,judge=True)
        ego_traj = self.query_single_trajs('ego')
        
        select_actor_ids = self.select_top_actors(self.obs_actors , self.walkers, [x_ego, y_ego])   
        neighbor_waypoints = np.zeros((6, 3, 50, 2))
        ego_waypoint = self.filter_and_pad([self.wp,self.wp2], [x_ego, y_ego])
        neighbor_waypoints[0] = ego_waypoint
        neighbor_trajs = np.zeros((6, 10, 5))
        neighbor_trajs[0] = ego_traj
        for i, actor_id in enumerate(select_actor_ids):
            actor_type = actor_id[1][1]
            index = actor_id[0]
            if actor_type==0:
                actor = self.obs_actors[index]
                neighbor_waypoints[i+1] = self.get_all_waypoints(actor)
                neighbor_trajs[i+1] = self.query_single_trajs('v_'+str(index))
            else:
                actor = self.walkers[index]
                neighbor_waypoints[i+1] = self.get_walker_waypoint(actor)
                neighbor_trajs[i+1] = self.query_single_trajs('w_'+str(index))
        
        neighbor_waypoints = neighbor_waypoints.reshape(18, 50, 2)
        return (neighbor_trajs, ego_traj[-1], neighbor_waypoints[:,::5])
    
    def action_adapter(self, model_action): 
        speed = model_action[0] # output (-1, 1)
        speed = (speed - (-1)) * (10 - 0) / (1 - (-1)) # scale to (0, 10) m/s
        
        speed = np.clip(speed, 0, 10)
        model_action[1] = np.clip(model_action[1], -1, 1)

        # discretization
        if model_action[1] < -1/3:
            lane = -1
        elif model_action[1] > 1/3:
            lane = 1
        else:
            lane = 0

        return (speed * 3.6, lane)
    
    def step(self, action):
        ## configurate the control command for the ego vehicle (if necessary)
        vx_ego = self.ego_vehicle.get_velocity().x
        vy_ego = self.ego_vehicle.get_velocity().y
        velocity_ego = (vx_ego**2 + vy_ego**2
                        + (self.ego_vehicle.get_velocity().z)**2)**(1/2)
        y_ego = self.ego_vehicle.get_location().y
        x_ego = self.ego_vehicle.get_location().x
        acceleration_ego = ((self.ego_vehicle.get_acceleration().x)**2 + (self.ego_vehicle.get_acceleration().y)**2
                        + (self.ego_vehicle.get_acceleration().z)**2)**(1/2)

        self.y_ego = y_ego
        self.x_ego = x_ego
        self.acceleration_ego = acceleration_ego
        self.vx_ego = vx_ego
        self.vy_ego = vy_ego
        self.velocity_ego = velocity_ego
    
        self.world.tick()
        
        waypoint = self.map.get_waypoint(self.ego_vehicle.get_location()) 
        target_speed, lat_action = self.action_adapter(action)

        self.target_speed = target_speed

        self.agent.set_target_speed(self.target_speed)

        preview_dis = round(np.clip(velocity_ego*2, 1, 15))
        wp_list = self.filter_planned_ego_waypoints(self.ego_vehicle, preview_dis)

        lr, r, rr = wp_list
        
        ## ego vehicle's lateral plan
        if x_ego > -30:
            lat_action = 0
        try:
            preview_dis = round(np.clip(velocity_ego*2, 1, 15)) 
            if lat_action == -1:
                if (waypoint.lane_change & carla.LaneChange.Left != 0) and (lr is not None):
                    target_location = waypoint.get_left_lane().next(preview_dis)[0].transform.location
           
                    target_location.x  = lr[0]
                    target_location.y  = lr[1]
                    self.agent.set_destination(target_location)
            
                else:
                    target_location = waypoint.next(preview_dis)[0].transform.location
                    target_location.x  = r[0]
                    target_location.y  = r[1]
                    self.agent.set_destination(target_location)
            elif lat_action == 1:
                if (waypoint.lane_change & carla.LaneChange.Right != 0)and (rr is not None):
                    target_location = waypoint.get_right_lane().next(preview_dis)[0].transform.location
                    target_location.x  = rr[0]
                    target_location.y  = rr[1]
                    self.agent.set_destination(target_location)
                else:
                    target_location = waypoint.next(preview_dis)[0].transform.location
                    target_location.x  = r[0]
                    target_location.y  = r[1]
                    self.agent.set_destination(target_location)
            else:
                target_location = waypoint.next(preview_dis)[0].transform.location
                target_location.x  = r[0]
                target_location.y  = r[1]
                self.agent.set_destination(target_location)
        except:
            pass
        
        ## set target speeds of obs vehicles
        v_index=0
        for v in self.obs_actors:
            if v.get_speed_limit() > 80 and self.speed_limit_obs_flags[v_index]==0:
                self.traffic_manager.vehicle_percentage_speed_difference(v,np.random.randint(47,67))
                self.speed_limit_obs_flags[int(v_index)] = 1
            v_index += 1

        ## walkers control
        last_walker_location = np.zeros((self.surrounding_number_walker))
        for i in range(len(self.walker_list)):
            control_walker = self.walkers[i].get_control()
            control_walker.speed = float(self.walker_speed[i])
            control_walker.direction = self.walker_direction[i]
            if abs(self.walkers[i].get_location().x - last_walker_location[i]) < 0.0005: 
                control_walker.jump = True
            else:
                control_walker.jump = False
            self.walkers[i].apply_control(control_walker)
            last_walker_location[i] = self.walkers[i].get_location().x
        
        self.control = self.agent.run_step()
    
        ## achieve the control to the ego vehicle
        self.ego_vehicle.apply_control(self.control)
        
        ## obtain the state transition and other variables after taking the action (control command)
        next_state = self.get_observation_scene()

        ## detect if the step is the terminated step, by considering: collision and episode fininsh
        self.collision = self.get_collision_history()[1]
        self.finish = (y_ego > -32) and (x_ego > -54 and x_ego <-50.5)
        self.max_time = self.count > 300
        
        success = 1 if self.finish else 0
        coll = -1 if self.collision else 0
        
        if self.finish or self.collision or self.off_route or self.max_time:
            done = True
        else:
            done = False
        
        reward = success + coll 

        info = (
            self.finish , self.collision , self.off_route , self.max_time
            )
        self.count += 1
        
        if done:
            self.destroy()

        return next_state, reward, done, info
    
    
    def judge_off_route(self, x, y, waypoint):
        min_list = []
        for wp in waypoint:
            dist = np.array(waypoint) - np.array([x, y])[np.newaxis,...]
            dist = np.linalg.norm(dist, axis=-1)
            min_list.append(np.min(dist))
        
        self.off_route = False if np.min(min_list) < 2 else True  
        if x < -55 or y < -70:
            self.off_route = True
        if (y > -25) and not (x > -54 and x <-50.5):
            self.off_route = True
        if (y<-66) and x<-30:
            self.off_route = True

    def destroy(self):

        self.collision_sensor.stop()
        actors = [
            self.ego_vehicle,
            self.collision_sensor,
            ]

        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors])
        
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.obs_list])
        
        self.collision_sensor = None
        self.ego_vehicle = None

    
    def IDM(self, index, y_target=None, v_target=None):
        delta = 4
        a, b, T, s0, v0 = 2.22, 1.67, 0.5, 5, 40

        # if there exists leading vehicle in the target lane
        if index is not None:
            
            s_delta = y_target - self.y_ego - 4
            v_delta = self.vy_ego - v_target
            s_prime = s0 + self.y_ego * T + self.vy_ego*v_delta/2/np.sqrt(a*b)
            
            acc = a * (1 - (self.vy_ego/v0)**delta - (s_prime/s_delta)**2)
            
            target_speed = self.target_speed + acc* (1/self.frame)
            target_speed = np.clip(target_speed, 36, 54)
        else:
        
            target_speed = v0
            
        return target_speed
        
    
    def _toggle_camera(self):
        self.camera_transform_index = (self.camera_transform_index + 1) % len(self.camera_transforms)
    
    
    def _next_sensor(self):
        self.camera_index += 1
        
        
    def _next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.world.set_weather(preset[0])
        

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

