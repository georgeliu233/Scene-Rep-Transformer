import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
layers = tf.keras.layers

class Rotater(tf.keras.layers.Layer):
    def __init__(self,name='rotater',mode='traj',aug=False, carla=False):
        super().__init__(name=name)
        self.mode = mode
        self.random_aug = aug
        self.carla = carla

    def call(self,states,mask,curr_frame=None,aug=True,random_angle=None):
        if self.mode=='traj':
            return self._make_rotations(states, mask,curr_frame,aug)
        elif self.mode=='fut':
            return self._rotate_fut(states,curr_frame,aug)
        else:
            return self._make_map_rotations(states, mask, curr_frame,aug,random_angle)

    def _rotate_fut(self,states,curr_frames,aug=True):
        '''
        input;[batch,timestep,2]
        curr_frame:[batch,5]
        '''
        mask = tf.not_equal(states, 0)[:,:,0]
        mask = tf.cast(mask,tf.float32)

        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1])
        #[batch,6,time_steps]
        x = states[:,:,0] - tf.reshape(curr_frames[:,0],[-1,1])
        y = states[:,:,1] - tf.reshape(curr_frames[:,1],[-1,1])

        new_x =   tf.multiply(cos_a,x) + tf.multiply(sin_a,y)
        new_y = - tf.multiply(sin_a,x) + tf.multiply(cos_a,y)

        rotated_state = [
            tf.expand_dims(new_x,2),
            tf.expand_dims(new_y,2)
        ]
        rotated_state =  tf.concat(rotated_state, axis=-1)
        mask = tf.expand_dims(mask, axis=-1)

        return tf.multiply(mask, rotated_state) , mask
        
    def _make_rotations(self,states,mask,curr_frame=None,aug=True):
        """
        ref : lhc_mtp_loss.py 
        make clock-wise rotation after moving to the ego point
        input states :[batch,ego+5*neighbours,timesteps,hidden]
        mask:[batch,ego+neighbors,timestep]

        """
        #get curr_frame location according to the mask
        mask = tf.cast(mask,tf.int32)
        ind = tf.reduce_sum(mask[:,0,:],axis=-1)
        ind = tf.clip_by_value(ind - 1,0,100)

        #gather indices is [batch_no , 0(ego) , mask_ind]
        #return [batch,hidden(4)]
        if curr_frame is None:
            curr_frames = tf.gather_nd(states, tf.transpose([
                tf.range(mask.get_shape()[0]), tf.zeros_like(ind) , ind,
            ]))
        else:
            #for future representations
            curr_frames = curr_frame
        if self.random_aug:
            r_range = np.pi/2
            random_angle =  tf.random.uniform(shape=curr_frames[:,2].get_shape(),minval=-r_range,maxval=r_range)
            cos_r = tf.reshape(tf.math.cos(random_angle),[-1,1,1])
            sin_r = tf.reshape(tf.math.sin(random_angle),[-1,1,1])
        

        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1,1])

        #[batch,6,time_steps]
        
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1])
        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])
        angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])

        if aug:
            new_x =   tf.multiply(cos_a,x) + tf.multiply(sin_a,y)
            new_y = - tf.multiply(sin_a,x) + tf.multiply(cos_a,y)
            if self.random_aug:
                n_x =   tf.multiply(cos_r,new_x) + tf.multiply(sin_r,new_y)
                n_y = - tf.multiply(sin_r,new_x) + tf.multiply(cos_r,new_y)
                new_x,new_y = n_x,n_y
        else:
            new_x,new_y = x,y

        vx = states[:,:,:,3] - tf.reshape(curr_frames[:,3],[-1,1,1])
        vy = states[:,:,:,4] - tf.reshape(curr_frames[:,4],[-1,1,1])
        if aug:
            new_vx =   tf.multiply(cos_a,vx) + tf.multiply(sin_a,vy)
            new_vy = - tf.multiply(sin_a,vx) + tf.multiply(cos_a,vy)
            if self.random_aug:
                n_vx =   tf.multiply(cos_r,new_vx) + tf.multiply(sin_r,new_vy)
                n_vy = - tf.multiply(sin_r,new_vx) + tf.multiply(cos_r,new_vy)
                new_vx,new_vy = n_vx,n_vy
        else:
            new_vx,new_vy = vx,vy

        rotated_state = [
            tf.expand_dims(-new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(angle,3),
            tf.expand_dims(-new_vx,3),
            tf.expand_dims(new_vy,3)
        ]
        if not self.carla:
            rotated_state = [
            tf.expand_dims(new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(angle,3),
            tf.expand_dims(new_vx,3),
            tf.expand_dims(new_vy,3)
        ]

        mask = tf.cast(tf.expand_dims(mask, axis=-1),tf.float32)
        rotated_state =  tf.concat(rotated_state, axis=-1)
        if self.random_aug:
            return tf.multiply(mask, rotated_state),curr_frames,random_angle
        return tf.multiply(mask, rotated_state),curr_frames,None
    
    def _make_map_rotations(self,states,mask,curr_frames,aug=True,random_angle=None):
        mask = tf.cast(mask,tf.int32)
        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1,1])
        # angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1])
        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])

        if self.random_aug:
            cos_r = tf.reshape(tf.math.cos(random_angle),[-1,1,1])
            sin_r = tf.reshape(tf.math.sin(random_angle),[-1,1,1])
        

        if aug:
            new_x =tf.multiply(cos_a,x) + tf.multiply(sin_a,y)
            new_y =-tf.multiply(sin_a,x) + tf.multiply(cos_a,y)
            if self.random_aug:
                n_x =   tf.multiply(cos_r,new_x) + tf.multiply(sin_r,new_y)
                n_y = - tf.multiply(sin_r,new_x) + tf.multiply(cos_r,new_y)
                new_x,new_y = n_x,n_y
        else:
            new_x,new_y = x,y
        rotated_state = [
            tf.expand_dims(-new_x,3),
            tf.expand_dims(new_y,3),
        ]
        if not self.carla:
            angle = states[:,:,:,2] - tf.reshape(yaw,[-1,1,1])
            rotated_state = [
                tf.expand_dims(new_x,3),
                tf.expand_dims(new_y,3),
                tf.expand_dims(angle,3),
                tf.expand_dims(states[:,:,:,3],3),
                tf.expand_dims(states[:,:,:,4],3)
            ]
        mask = tf.cast(tf.expand_dims(mask, axis=-1),tf.float32)
        rotated_state =  tf.concat(rotated_state, axis=-1)


        return tf.multiply(mask, rotated_state)

class Ego_Neighbours_Encoder(tf.keras.layers.Layer):
    def __init__(self, state_shape,name='ego_surr_encode',units=256,
        use_trans_encode=False,num_heads=6,drop_rate=0.1,neighbours=5,
        make_rotation=True,time_step=8,bptt=False):
        super().__init__(name=name)
        self.neighbours = neighbours
        self.make_rotation=make_rotation
        self.time_step=time_step
        self.lstm_layer = layers.GRU(units,return_sequences=False)
        self.use_trans = use_trans_encode
        self.ego_norm = layers.LayerNormalization()
        self.bptt = bptt
        if use_trans_encode:
            self.rel_layer = layers.MultiHeadAttention(num_heads,units//num_heads,dropout=drop_rate)
            self.FFN_layers = [
                layers.Dense(4*units,activation='elu'),
                layers.Dense(units)
            ]
            self.dropout_layers = [
                layers.Dropout(drop_rate)
            ]*len(self.FFN_layers)

            self.norm_layers = [
                layers.LayerNormalization()
            ]*2
        else:
            self.encode_layer = layers.Dense(
                units,activation='elu'
            )
    
    def wrap_to_pi(self,theta):
        pi = tf.constant(np.pi)
        return (theta+pi) % (2*pi) - pi

    def get_speed(self,inputs):
        pad = tf.expand_dims(tf.zeros_like(inputs[:,:,0]),2)
        speed = (inputs[:,:,1:] - inputs[:,:,:-1])/0.1
        return tf.concat([pad,speed], axis=-1)

    def _make_rotations(self,states,mask):
        """
        ref : lhc_mtp_loss.py 
        make clock-wise rotation after moving to the ego point
        input states :[batch,ego+5*neighbours,timesteps,hidden]

        """
        #get curr_frame location according to the mask
        ind = tf.reduce_sum(mask,axis=-1)

        #gather indices is [batch_no , 0(ego) , mask_ind]
        #return [batch,hidden(4)]
        curr_frames = tf.gather_nd(states, tf.transpose([
            tf.range(mask.get_shape()[0]), tf.zeros_like(ind) , ind-1
        ]))

        yaw = curr_frames[:,2]
        cos_a = tf.reshape(tf.math.cos(yaw),[-1,1,1])
        sin_a = tf.reshape(tf.math.sin(yaw),[-1,1,1])

        #[batch,6,time_steps]
        mask = tf.expand_dims(mask, axis=1)
        x = states[:,:,:,0] - tf.reshape(curr_frames[:,0],[-1,1,1])
        x = tf.multiply(tf.cast(mask,tf.float32),x)

        y = states[:,:,:,1] - tf.reshape(curr_frames[:,1],[-1,1,1])
        y = tf.multiply(tf.cast(mask,tf.float32),y)

        # vx = states[:,:,:,2] - tf.reshape(curr_frames[:,2],[-1,1,1])
        vx = tf.multiply(tf.cast(mask,tf.float32),states[:,:,:,3])

        # vy = states[:,:,:,3] - tf.reshape(curr_frames[:,3],[-1,1,1])
        vy = tf.multiply(tf.cast(mask,tf.float32),states[:,:,:,4])

        # dis = states[:,:,:,4] - tf.reshape(curr_frames[:,4],[-1,1,1])
        angle = tf.multiply(tf.cast(mask,tf.float32),self.wrap_to_pi(states[:,:,:,5]-tf.reshape(yaw,[-1,1,1])))
        new_x =tf.multiply(cos_a,x) - tf.multiply(sin_a,y)
        new_y =tf.multiply(sin_a,x) + tf.multiply(cos_a,y)

        new_vx =tf.multiply(cos_a,vx) - tf.multiply(sin_a,vy)
        new_vy =tf.multiply(sin_a,vx) + tf.multiply(cos_a,vy)

        rotated_state = [
            tf.expand_dims(new_x,3),
            tf.expand_dims(new_y,3),
            tf.expand_dims(new_vx,3),
            tf.expand_dims(new_vy,3),
            tf.expand_dims(angle,3)
        ]
        return tf.concat(rotated_state, axis=-1)

    def _split_ego(self,states,mask):

        if self.make_rotation:
            states = self._make_rotations(states,mask)
        else:
            x = states[:,:,:,0]
            y = states[:,:,:,1]
            v = states[:,:,:,2]
            dis = states[:,:,:,3]
            angle = self.wrap_to_pi(states[:,:,:,4])
            rotated_state = [
            tf.expand_dims(x,3),
            tf.expand_dims(y,3),
            tf.expand_dims(v,3),
            tf.expand_dims(dis,3),
            tf.expand_dims(angle,3)
            ]
            states = tf.concat(rotated_state, axis=-1)

        e_states,n_states = states[:,0,:,:] , states[:,1:,:,:]
        
        return e_states , n_states

    def _lstm_with_mask(self,states,mask,return_hidden=False,init_state=None):
        if return_hidden:
            full_states = self.lstm_layer(inputs=states,mask=tf.cast(mask, tf.bool),initial_state=tf.cast(init_state,tf.float32))
        else:
            full_states = self.lstm_layer(inputs=states,mask=tf.cast(mask, tf.bool))
        ind = tf.reduce_sum(mask,axis=-1)
        states = tf.gather_nd(full_states, tf.transpose([
            tf.range(mask.get_shape()[0]),ind-1
        ]))
        if return_hidden:
            return states , full_states
        return states

    def call(self,states,mask=None,test=False,debug=False,init_state=None):
        mask = tf.not_equal(states[:,:,:,0],0)
        mask = tf.cast(mask,tf.int32)
        training= bool(1-test)
        ego_states , neighbor_states = self._split_ego(states,mask)
        actor_mask = tf.not_equal(tf.concat([tf.expand_dims(tf.ones_like(ego_states), 1), neighbor_states], axis=1), 0)[:, :, 0, 0]
        if self.bptt:
            ego,full_ego = self._lstm_with_mask(ego_states,mask,return_hidden=True,init_state=init_state)
        else:
            ego= self.lstm_layer(ego_states)
        neighbors =[
            self.lstm_layer(neighbor_states[:,i,:,:])for i in range(5)
        ]
        neighbors = tf.stack(neighbors, axis=1)
        actor = tf.concat([ego[:, tf.newaxis], neighbors], axis=1)
        if self.use_trans:
            value = self.rel_layer(tf.expand_dims(ego, axis=1), actor,attention_mask=actor_mask[:, tf.newaxis], training=training)
            value = tf.squeeze(value,axis=1)
            value = self.norm_layers[0](value)
            for i in range(len(self.FFN_layers)):
                value = self.FFN_layers[i](value)
                value = self.dropout_layers[i](value,training=training)
            value = self.norm_layers[-1](value)
        else:
            value = self.encode_layer(actor)
        feature = tf.concat([value,self.ego_norm(ego)],axis=-1)    
        return feature
    

class MapEncoder(tf.keras.layers.Layer):
    def __init__(self,return_attention_scores=False,carla=False):
        super(MapEncoder, self).__init__()
        self.return_attention_scores = return_attention_scores
        # self.node_feature = tf.keras.layers.Conv1D(64, 1, activation='elu')
        self.node_attention = tf.keras.layers.MultiHeadAttention(2, 128, dropout=0, output_shape=64*3)
        self.flatten = tf.keras.layers.GlobalMaxPooling1D()
        self.vector_feature = tf.keras.layers.Dense(64, activation='relu')
        self.sublayer = tf.keras.layers.Dense(128, activation='relu')
        self.carla = carla

    def call(self, inputs, mask,test):
        mask = tf.cast(mask, tf.int16)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        if self.carla:
            nodes = inputs[:, :, :2]
        else:
            nodes = inputs[:, :, :3]
        if self.return_attention_scores:
            nodes,val = self.node_attention(nodes, nodes, attention_mask=mask,training=bool(1-test),return_attention_scores=self.return_attention_scores)
        else:
            nodes = self.node_attention(nodes, nodes, attention_mask=mask,training=bool(1-test),return_attention_scores=self.return_attention_scores)
        nodes = tf.nn.relu(nodes)
        nodes = self.flatten(nodes)
        vector = self.vector_feature(inputs[:, 0, -2:])
        out = tf.concat([nodes, vector], axis=1)
        polyline_feature = self.sublayer(out)
        if self.return_attention_scores:
            # print(val.get_shape())
            val = tf.reduce_mean(tf.reduce_mean(val,axis=1),axis=1)
            return polyline_feature,val
        return polyline_feature

class MultiModal_Attention(tf.keras.layers.Layer):
    def __init__(self, num_modes, key_dim,head_num=1):
        super(MultiModal_Attention, self).__init__()
        self._num_modes = num_modes
        self.attention = [tf.keras.layers.MultiHeadAttention(head_num, key_dim, dropout=0,output_shape=key_dim) for _ in range(num_modes)]
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        # self.FFN1 = tf.keras.layers.Dense(4*key_dim, activation='elu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.FFN2 = tf.keras.layers.Dense(key_dim, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.1)        

    def call(self, query, key, mask=None, training=True):
        output = []
        for i in range(self._num_modes):
            value= self.attention[i](query, key, attention_mask=mask, return_attention_scores=False, training=training)
            # print(value.get_shape())
            output.append(tf.squeeze(value,axis=1))  
        value = tf.nn.relu(tf.stack(output, axis=1))
        return value, None

class Hierachial_Transformer(tf.keras.Model):
    def __init__(self, state_shape,name='hier_encoder',units=256,
        use_trans_encode=False,num_heads=6,drop_rate=0.1,neighbours=5,
        make_rotation=True,time_step=8,bptt=False,num_modes=1,final_head_num=2,
        random_aug=False,no_ego_fut=False,no_neighbor_fut=False,carla=False):
        super(Hierachial_Transformer, self).__init__()

        self.map_layer = MapEncoder(return_attention_scores=True,carla=carla)

        self.neighbours = neighbours
        self.make_rotation=make_rotation
        self.time_step=time_step
        # self.embed_layer = layers.Dense(units)
        # self.time_embed_layer = layers.Dense(units)
        self.time_layer = layers.MultiHeadAttention(num_heads,units,dropout=0,output_shape=units)
        self.time_pooling = layers.GlobalMaxPooling1D()

        self.use_trans = use_trans_encode
        # self.ego_norm = layers.LayerNormalization()
        self.bptt = bptt
        self.rel_layer = layers.MultiHeadAttention(num_heads,units,dropout=0,output_shape=units)
        if self.make_rotation:
            self.rotater = Rotater(mode='traj',aug=random_aug,carla=carla)
            self.map_rotater = Rotater(mode='map',aug=random_aug,carla=carla)

        self.map_attention = tf.keras.layers.MultiHeadAttention(num_heads, units, dropout=0,output_shape=units) 
        #wenti
        self.final_attention = [tf.keras.layers.MultiHeadAttention(final_head_num, units, dropout=0,output_shape=units) for _ in range(num_modes)]
        # self.final_attention = [tf.keras.layers.MultiHeadAttention(2, units, dropout=0,output_shape=units) for _ in range(num_modes)]
        self.num_modes = num_modes

        self.no_ego_fut = no_ego_fut
        self.no_neighbor_fur = no_neighbor_fut

        dummy_state = tf.constant(np.zeros(shape=(32,) + state_shape, dtype=np.float32))
        self.carla = carla
        if carla:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*3,time_step,2), dtype=np.float32))
        else:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*2,time_step,5), dtype=np.float32))

        self(dummy_state,map_state=map_s)
        self.summary()
        

    def call(self,states,test=False,map_state=None,aug=True):

        training= bool(1-test)
        mask = tf.not_equal(states, 0)[:,:,:,0]
        if self.make_rotation:
            states,curr_frames,rg = self.rotater(states,mask,aug=aug)

        ego_states , neighbor_states = states[:,0,:,:] , states[:,1:,:,:]
        ego_mask,neighbor_mask = mask[:,0,:],  mask[:,1:,:]

        actor_mask = tf.not_equal(tf.concat([tf.expand_dims(tf.ones_like(ego_states), 1), neighbor_states], axis=1), 0)[:, :, 0, 0]
        ego = self._timestep_attention(ego_states,training,ego_mask)
        neighbors =[
            self._timestep_attention(neighbor_states[:,i,:,:],training,neighbor_mask[:,i,:]) for i in range(self.neighbours)
        ]

        map_mask = tf.not_equal(map_state, 0)[:,:,:,0]
        map_traj_mask = tf.not_equal(map_state, 0)[:,:,0,0]
        
        if self.make_rotation:
            map_state = self.map_rotater(map_state,map_mask,curr_frames,aug,rg)
        

        map = [self.map_layer(map_state[:, i], map_mask[:, i, :],test)[0] for i in range(map_state.get_shape().as_list()[1])]
        if test:
            val = [self.map_layer(map_state[:, i], map_mask[:, i, :],test)[1] for i in range(map_state.get_shape().as_list()[1])]
            val = tf.stack(val,axis=1)
        #(b,12,256)
        map = tf.stack(map, axis=1)
        
        if self.carla:
            ego_map,neighbor_map = map[:,:3,:],map[:,3:,:]
            ego_map_traj_mask,neighbor_map_traj_mask = map_traj_mask[:,:3],map_traj_mask[:,3:]
        else:
            ego_map,neighbor_map = map[:,:2,:],map[:,2:,:]
            ego_map_traj_mask,neighbor_map_traj_mask = map_traj_mask[:,:2],map_traj_mask[:,2:]
        # ego_map_mask,neighbor_map_mask = map_mask[:,:2,:],map_mask[:,2:,:]

        neighbor_rel_val = [
            self._map_vehicle_rel(neighbors[i],neighbor_map,neighbor_map_traj_mask,i*2)[0] for i in range(self.neighbours)
        ]
        if self.no_neighbor_fur:
            neighbor_rel_val = neighbors
        if True:
            neighbor_val = [
                self._map_vehicle_rel(neighbors[i],neighbor_map,neighbor_map_traj_mask,i*2)[1] for i in range(self.neighbours)
            ]

        actor = tf.concat([ego[:, tf.newaxis], tf.stack(neighbor_rel_val, axis=1)], axis=1)
        actor_rel = self.rel_layer(tf.expand_dims(ego, axis=1), actor,attention_mask=actor_mask[:, tf.newaxis], training=training)
        actor_rel = tf.nn.relu(tf.squeeze(actor_rel,axis=1))

        goals,ego_val = self._goal_layer(actor_rel[:,tf.newaxis], ego_map, ego_map_traj_mask[:,tf.newaxis])
        # ego_states = tf.concat([actor_rel, ego], axis=-1)
        ego_states = tf.repeat(actor_rel[:, tf.newaxis], self.num_modes, axis=1)

        if self.no_ego_fut:
            states = ego_states
        else:
            states = goals + ego_states #+ ego
        if test:
            neighbor_val = [ego_val] + neighbor_val
            neighbor_val = tf.expand_dims(tf.concat(neighbor_val,axis=-1),axis=-1)
            return states,neighbor_val #tf.multiply(neighbor_val,val)

        return states

    def _timestep_attention(self,states,training,mask):
        #[batch,timesteps,dim]
        mask = tf.cast(mask, tf.int16)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        state_val = self.time_layer(states,states,attention_mask=mask,training=training)
        state_val = tf.nn.relu(state_val)
        state_val = self.time_pooling(state_val)
        return state_val
    
    def _map_vehicle_rel(self,value,map_state,map_mask,i):
        use_map = map_state[:,i:i+2,:]
        use_map_mask = tf.concat([tf.ones_like(map_mask[:,0])[:,tf.newaxis],map_mask[:,i:i+2]],axis=1)[:,tf.newaxis]
        mv_rel = tf.concat([value[:, tf.newaxis], use_map], axis=1)
        mv_val,val = self.map_attention(value[:,tf.newaxis],mv_rel,attention_mask=use_map_mask,training=True,return_attention_scores=True)
        val = tf.reduce_mean(tf.squeeze(val,axis=-2),axis=1)[:,1:]
        # print(val.get_shape())
        mv_val = tf.squeeze(mv_val,axis=1)
        mv_val = tf.nn.relu(mv_val)
        return mv_val,val
    
    def _goal_layer(self,query, key, mask=None, training=True):
        output,v = [],[]
        for i in range(self.num_modes):
            value,val= self.final_attention[i](query, key, attention_mask=mask, return_attention_scores=True,training=training)
            output.append(tf.squeeze(value,axis=1))
            v.append(val)
        v = tf.reduce_mean(tf.squeeze(v[0],axis=-2),axis=1)
        value = tf.nn.relu(tf.stack(output, axis=1))
        return value,v
    
class GuassianActor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6
    def __init__(self,params,state_shape,action_dim, max_action,hidden_activation="relu", name='gaussian_policy'
        ,state_independent_std=False,squash=True,drop_rate=0.1,ensembles=True):
        super().__init__(name=name)

        if params['make_prediction']:
            head_num=params['head_num']
        else:
            head_num=params['head_num']
        self.encoder = RLEncoder(state_shape, action_dim,units=[256]*3,hidden_activation=hidden_activation,
                    state_input=params['state_input'],lstm=params['LSTM'],trans=False,
                    cnn_lstm=params['cnn_lstm'],bptt=params['bptt'],ego_surr=params['ego_surr'],
                    use_trans=params['use_trans'],neighbours=params['neighbours'],time_step=params['time_step'],debug=False,
                    make_rotation=params['make_rotation'],make_prediction=params['make_prediction'],use_map=params['use_map']
                    ,num_traj=params['traj_nums'],path_length=params['path_length'],head_dim=head_num,cnn=params['cnn'],
                    carla=params['carla'])
        
        self._state_independent_std = state_independent_std
        self._squash = squash
        self.use_map=params['use_map']
        self.params = params
        self.ensemble = ensembles
        self.multi_selection = 'multi_selection' in list(params.keys())

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action

        self.actor_layers = [layers.Dense(128, activation=hidden_activation),
                            layers.Dense(32, activation=hidden_activation)]
        
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        self.make_prediction = params['make_prediction']
            
        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))
        
        if params['use_map']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state,mask=m,init_state=init_state,map_state=map_s)
        self.summary()
    
    def call(self,states,mask=None, test=False,init_state=None,map_state=None):

        features,h = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)

        for layer in self.actor_layers:
            features = layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)
            std = tf.exp(log_std)
        
        if self.use_map:
            mean = tf.reduce_mean(mean, axis=1)
            std = tf.reduce_mean(std, axis=1)

        dist =  tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)
        entropy = dist.entropy()

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=-1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action

        return actions,log_pis,entropy,h


class ValueNet(tf.keras.Model):
    def __init__(self,params,state_shape,action_dim,drop_rate=0.1,hidden_activation="relu", name='value_network'):
        super().__init__(name=name)

        if params['make_prediction'] and params['make_prediction_value']:
            head_num=1
            num_traj = params['traj_nums']
        else:
            head_num=params['head_num']
            num_traj = 1
            
        self.encoder = RLEncoder(state_shape, action_dim,units=[256]*3,hidden_activation=hidden_activation,
                    state_input=params['state_input'],lstm=params['LSTM'],trans=False,
                    cnn_lstm=params['cnn_lstm'],bptt=params['bptt'],ego_surr=params['ego_surr'],
                    use_trans=params['use_trans'],neighbours=params['neighbours'],time_step=params['time_step'],debug=False,
                    make_rotation=params['make_rotation'],make_prediction=params['make_prediction'],use_map=params['use_map']
                    ,num_traj=num_traj,path_length=params['path_length'],head_dim=head_num,cnn=params['cnn'],carla=params['carla'])
        
        self.make_prediction = params['make_prediction_value']
        self.use_map=params['use_map']

        self.critic_layers = [ 
                                layers.Dense(128, activation=hidden_activation), 
                                layers.Dense(32, activation=hidden_activation),
                                layers.Dense(1)]

        self.params = params
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state,mask=m,init_state=init_state,map_state=map_s)
        self.summary()
    
    def call(self,states,mask=None, test=False,init_state=None,map_state=None):
        features,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)
        for layer in self.critic_layers:
            features = layer(features)
        if self.use_map:
            features = tf.reduce_mean(features,axis=1)
        values = tf.squeeze(features, axis=1)
        return values



class QNet(tf.keras.Model):
    def __init__(self,params,state_shape,action_dim,hidden_activation="relu", name='q_network'):
        super().__init__(name=name)

        if params['make_prediction'] and params['make_prediction_q']:
            head_num=1
            num_traj = params['traj_nums']
        else:
            head_num=params['head_num']
            num_traj = 1
        self.encoder = RLEncoder(state_shape, action_dim,units=[256]*3,hidden_activation=hidden_activation,
                    state_input=params['state_input'],lstm=params['LSTM'],trans=False,
                    cnn_lstm=params['cnn_lstm'],bptt=params['bptt'],ego_surr=params['ego_surr'],
                    use_trans=params['use_trans'],neighbours=params['neighbours'],time_step=params['time_step'],debug=False,
                    make_rotation=params['make_rotation'],make_prediction=params['make_prediction'],use_map=params['use_map']
                    ,num_traj=num_traj,path_length=params['path_length'],head_dim=head_num,cnn=params['cnn'],carla=params['carla']
                    )
        
        self.use_map=params['use_map']
        self.critic_layers = [layers.Dense(128, activation=hidden_activation), 
                                layers.Dense(32, activation=hidden_activation),
                                layers.Dense(1)]
        
        self.action_layer = [layers.Dense(64, activation=hidden_activation)]
        self.make_prediction = params['make_prediction_q']
        self.multi_selection = 'multi_selection' in list(params.keys())
        
        self.traj_nums = params['traj_nums']
        self.traj_length = params['traj_length']

        self.params = params
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state,dummy_action,mask=m,init_state=init_state,map_state=map_s)
        self.summary()
    
    def call(self,states,actions,mask=None, test=False,init_state=None,map_state=None,traj=None):

        features,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)
        if self.use_map:
            ensembles = features.get_shape()[1]
            act_feature = tf.expand_dims(self.action_layer[0](actions),1)
            features =  tf.concat([features, tf.repeat(act_feature,ensembles,axis=1)], axis=2)
        else:
            act_feature = self.action_layer[0](actions)
            features =  tf.concat([features, act_feature], axis=1)

        for layer in self.critic_layers:
            features = layer(features)
        if self.use_map:
            features = tf.reduce_mean(features,axis=1)
        values = tf.squeeze(features, axis=-1)
        return values


class RLEncoder(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=[256]*3,
                hidden_activation="relu",name='rl_encoder',state_input=False,lstm=False,trans=False,
                cnn_lstm=False,bptt=False,ego_surr=False,use_trans=False,neighbours=5,time_step=8,debug=False,
                make_rotation=True,make_prediction=False,use_mask=False,use_map=False,num_traj=5
                ,cnn=False,path_length=0,head_dim=1,use_hier=False,random_aug=False,no_ego_fut=False,no_neighbor_fut=False,
                carla=False):
        super().__init__(name=name)

        self.lstm = lstm
        self.cnn = cnn
        self.cnn_lstm = cnn_lstm
        self.state_input = state_input
        self.bptt=bptt
        self.ego_surr=ego_surr
        
        self.trans = trans
        self.debug=debug
        self.use_map=use_map
        self.neighbours = neighbours
        self.num_traj = num_traj

        self.use_mask=use_mask
        self.use_hier = use_hier

        if cnn:
            #CNN
            print('using cnn')
            self.encode_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                layers.GlobalAveragePooling2D()]
        elif lstm and state_input:
            self.lstm_layers = layers.GRU(256,return_sequences=True)
            print('lstm')
            self.encode_layers = [
                layers.Dense(256,activation=hidden_activation)]
        elif cnn_lstm:
            print('using cnn lstm')
            self.cnn_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                layers.GlobalAveragePooling2D()]
            self.lstm_layers = layers.GRU(256,return_sequences=True)
            self.encode_layers = [layers.Dense(256,activation='relu')]   
        elif ego_surr:
            print('using ego surrounding encoder')
            self.ego_layer = Ego_Neighbours_Encoder(state_shape,use_trans_encode=use_trans,
            neighbours=neighbours,time_step=time_step,num_heads=6,bptt=bptt,make_rotation=make_rotation)
        elif use_map:
            print('using map encoder')
            self.ego_layer = Ego_Neighbours_Encoder(state_shape,use_trans_encode=True,
            neighbours=neighbours,time_step=time_step,num_heads=2,bptt=bptt,make_rotation=make_rotation,
            units=128,random_aug=random_aug)
            self.map_layer = MapEncoder()
            self.goal_layer = MultiModal_Attention(num_traj,128,head_dim)
            self.norm = layers.LayerNormalization()
            self.frame_layer = layers.Dense(64,activation='relu')
            self.make_rotation = make_rotation
   
        elif use_hier:
            print('use Hiereachial Transformer')
            self.h_layer = Hierachial_Transformer(state_shape,units=128,use_trans_encode=True,
            num_heads=2,drop_rate=0, neighbours=neighbours, make_rotation=make_rotation, time_step=time_step
            , bptt=False, num_modes=num_traj,final_head_num=head_dim,random_aug=random_aug,
            no_ego_fut=no_ego_fut,no_neighbor_fut=no_neighbor_fut,carla=carla)
        else:
            print('using mlp state')
            self.encode_layers = []
            for unit in units:
                self.encode_layers.append(layers.Dense(unit, activation=hidden_activation))

        dummy_state = tf.constant(np.zeros(shape=(32,) + state_shape, dtype=np.float32))
        mask = tf.ones([32,dummy_state.get_shape()[1]])

        if not bptt:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((32,256))

        if use_map or use_hier:
            map_s = tf.constant(np.zeros(shape=(32,) + (state_shape[0]*2,path_length,5), dtype=np.float32))
        else:
            map_s = None

        self(dummy_state,mask=m,init_state=init_state,map_state=map_s)
        self.summary()
    
    def _thru_cnn(self,states):
        for layer in self.cnn_layers:
            states = layer(states)
        return states
    
    def call(self,states,mask,test=False,init_state=None,map_state=None,curr_frames=None,aug=True):
        if self.cnn and not aug:
            # augmentations for DrQ
            states = tf.image.random_crop(value=states, size=states.get_shape())
        if self.ego_surr:
            if self.bptt:
                states,full_states,ego,_ = self.ego_layer(states,mask,test,init_state=init_state)
            else:
                states,ego,_ = self.ego_layer(states,mask,test)
        elif self.use_hier:
            if test:
                # print(states.get_shape(),map_state.get_shape())
                states,val = self.h_layer(states,test,map_state,aug)
                return states,val
            states = self.h_layer(states,test,map_state,aug)
        elif self.use_map:
            if self.bptt:
                states,full_states,ego,curr_frame ,rg= self.ego_layer(states,mask,test,init_state=init_state,curr_frame=curr_frames,
                aug=aug)
            else:
                states,ego,curr_frame,rg = self.ego_layer(states,mask,test,curr_frame=curr_frames,aug=aug)

            map_mask = tf.not_equal(map_state, 0)[:,:,:,0]
            if self.make_rotation:
                map_state = self.rotater(map_state,map_mask,curr_frame,aug,rg)

            map = [self.map_layer(map_state[:, i], map_mask[:, i, :],test) for i in range(map_state.get_shape().as_list()[1])] 

            #(12,256)
            map = tf.stack(map, axis=1)
            #find potential goals
            goals, _ = self.goal_layer(tf.expand_dims(states, axis=1),map,map_mask[:,:,0][:, tf.newaxis])  

            ego_states = tf.concat([states, ego], axis=-1)
            ego_states = tf.repeat(ego_states[:, tf.newaxis], self.num_traj, axis=1)

            states = tf.concat([goals, ego_states], axis=-1)

            # states = self.out_layer(states)
        elif self.cnn_lstm:
            new_state = []
            for i in range(states.get_shape()[1]):
                new_state.append(self._thru_cnn(states[:,i,:,:,:]))
            states = tf.stack(new_state,axis=1)
            mask = tf.cast(mask,tf.int32)
            if self.bptt:
                full_states= self.lstm_layers(inputs=tf.cast(states,tf.float32),mask=tf.cast(mask, tf.bool),initial_state=tf.cast(init_state,tf.float32))
            else:
                full_states= self.lstm_layers(inputs=tf.cast(states,tf.float32),mask=tf.cast(mask, tf.bool))
            states = full_states[:,-1]
        else:
            if self.lstm:
                mask = tf.cast(mask,tf.int32)
                if self.bptt:
                    full_states = self.lstm_layers(inputs=tf.cast(states,tf.float32),mask=tf.cast(mask, tf.bool),initial_state=tf.cast(init_state,tf.float32))
                else:
                    full_states = self.lstm_layers(inputs=tf.cast(states,tf.float32),mask=tf.cast(mask, tf.bool))
                ind = tf.reduce_sum(mask,axis=-1)
                states = full_states[:,-1]
          
            for layer in self.encode_layers:
                states = layer(states)

        if self.bptt and (self.cnn_lstm or self.lstm or self.ego_surr):
            return states,full_states
            # print(h,c)
        if self.use_map:
            return states,curr_frame
        
        return states,None
