
import tensorflow_probability as tfp
from sac_actor_critic_policy import RLEncoder
from tf2rl.misc.target_update_ops import update_target_variables
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers

class SAC_Critic(tf.keras.Model):
    def __init__(self,params,state_shape,action_dim,hidden_activation="relu", name='actor_critic_network'):
        super().__init__(name=name)

        if params['make_prediction']:
            head_num=1#params['head_num']
        else:
            head_num=params['head_num']
        self.encoder = RLEncoder(state_shape, action_dim,units=[256]*3,hidden_activation=hidden_activation,
                    state_input=params['state_input'],lstm=params['LSTM'],trans=False,
                    cnn_lstm=params['cnn_lstm'],bptt=params['bptt'],ego_surr=params['ego_surr'],
                    use_trans=params['use_trans'],neighbours=params['neighbours'],time_step=params['time_step'],debug=False,
                    make_rotation=params['make_rotation'],make_prediction=params['make_prediction'],use_map=params['use_map']
                    ,num_traj=params['traj_nums'],path_length=params['path_length'],head_dim=head_num,cnn=params['cnn'],
                    use_hier=params['use_hier'],random_aug=params['random_aug'],
                    no_ego_fut=params['no_ego_fut'],no_neighbor_fut=params['no_neighbor_fut'])
        self.params = params

        self.multi_selection = 'multi_selection' in list(params.keys())
        
        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,dummy_state.get_shape()[1]])

        self.use_map= params['use_map'] or params['use_hier']
        if 'ensembles' not in params:
            self.critic_layers = [layers.Dense(128, activation=hidden_activation), 
                                    layers.Dense(32, activation=hidden_activation),
                                    layers.Dense(1)]
            
            self.action_layer = [layers.Dense(64, activation=hidden_activation)]

            self.critic_layers_2 = [layers.Dense(128, activation=hidden_activation), 
                                    layers.Dense(32, activation=hidden_activation),
                                    layers.Dense(1)]
            
            self.action_layer_2 = [layers.Dense(64, activation=hidden_activation)]
        else:
            # print('en-critic')
            self.critic_layers = []
            self.critic_layers_2 = []
            self.action_layer = []
            self.action_layer_2 = []
            for i in range(params['traj_nums']):
                self.critic_layers.append([layers.Dense(128, activation=hidden_activation,name=f"c_0_{i}"), 
                                        layers.Dense(32, activation=hidden_activation,name=f"c_1_{i}"),
                                        layers.Dense(1,name=f"c_2_{i}")])
                self.critic_layers_2.append([layers.Dense(128, activation=hidden_activation,name=f"c2_0_{i}"), 
                                        layers.Dense(32, activation=hidden_activation,name=f"c2_1_{i}"),
                                        layers.Dense(1,name=f"c2_2_{i}")])
                
                self.action_layer.append(layers.Dense(64, activation=hidden_activation,name=f"action_0_{i}"))
                self.action_layer_2.append(layers.Dense(64, activation=hidden_activation,name=f"action_2_0_{i}"))

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map'] or params['use_hier']:
            map_s = tf.constant(np.zeros(shape=(1,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
        else:
            map_s = None
        
        self.make_prediction = params['make_prediction']
        if self.make_prediction:
            self.traj_nums = params['traj_nums']
            self.traj_length = params['traj_length']
            self.prediction_layers = [
                                layers.Dense(128, activation=hidden_activation,name='pred_0'),
                                layers.Dense(64, activation=hidden_activation,name='pred_1'),
                                layers.Dense(self.traj_length*2,name='pred_layer')]

        self(dummy_state,dummy_action,mask=m,init_state=init_state,map_state=map_s)
        h = self._get_hidden(dummy_state,mask=m,init_state=init_state,map_state=map_s)
        self.h_size = tuple(h.get_shape().as_list())
        self.summary()

    def call(self,states,actions,mask=None,test=False,init_state=None,map_state=None,traj=None,curr_frames=None,en_input=False,aug=True):

        feat,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state,aug=aug)

        # feat = feat[:,:,128:]
        
        if 'ensembles' in self.params:
            res_value = []
            res_value_2 = []
            for i in range(self.params['traj_nums']):
                if en_input:
                    act_feature = self.action_layer[i](actions[:,i])
                    act_feature_2 = self.action_layer_2[i](actions[:,i])
                else:
                    act_feature = self.action_layer[i](actions)
                    act_feature_2 = self.action_layer_2[i](actions)

                features =  tf.concat([feat[:,i], act_feature], axis=1)
                features_2 =  tf.concat([feat[:,i], act_feature_2], axis=1)

                for layer,layer_2 in zip(self.critic_layers[i],self.critic_layers_2[i]):
                    features = layer(features)
                    features_2 = layer_2(features_2)
              
                res_value.append(features)
                res_value_2.append(features_2)
            # print(tf.stack(res_value,axis=1).get_shape())
            return tf.squeeze(tf.stack(res_value,axis=1),axis=-1) , tf.squeeze(tf.stack(res_value_2,axis=1),axis=-1)

        if self.use_map:
            ensembles = feat.get_shape()[1]
            act_feature = tf.expand_dims(self.action_layer[0](actions),1)
            features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=-1)

            act_feature_2 = tf.expand_dims(self.action_layer_2[0](actions),1)
            features_2 =  tf.concat([feat, tf.repeat(act_feature_2,ensembles,axis=1)], axis=-1)
            if self.multi_selection:
                raise NotImplementedError('multi selection have not been implemented!')
        else:
            act_feature = self.action_layer[0](actions)
            features =  tf.concat([feat, act_feature], axis=1)

            act_feature_2 = self.action_layer_2[0](actions)
            features_2 =  tf.concat([feat, act_feature_2], axis=1)


        for layer in self.critic_layers:
            features = layer(features)

        for layer in self.critic_layers_2:
            features_2 = layer(features_2)
        if self.use_map:
            features = tf.reduce_mean(features,axis=1)
            features_2 = tf.reduce_mean(features_2,axis=1)

        values = tf.squeeze(features, axis=-1)
        values_2 = tf.squeeze(features_2, axis=-1)
        return values,values_2
    
    def _get_hidden(self,states,mask=None, test=False,init_state=None,map_state=None,aug=True):
        features,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state,aug=aug)
        return features#[:,:,128:]
    
    def _get_hidden_and_val(self,states,mask=None, test=True,init_state=None,map_state=None,aug=True):
        features,val = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state,aug=aug)
        return features,val

    def _get_pca_val(self,states,actions,mask=None, test=True,init_state=None,map_state=None,aug=True):
        feat,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state,aug=aug)
        ensembles = feat.get_shape()[1]
        act_feature = tf.expand_dims(self.action_layer[0](actions),1)
        features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=-1)

        act_feature_2 = tf.expand_dims(self.action_layer_2[0](actions),1)
        features_2 =  tf.concat([feat, tf.repeat(act_feature_2,ensembles,axis=1)], axis=-1)

        out_feat = 0.5*(features + features_2)
        out_feat=tf.squeeze(out_feat,axis=1)

        for layer in self.critic_layers:
            features = layer(features)

        for layer in self.critic_layers_2:
            features_2 = layer(features_2)
        
        values = tf.squeeze(features, axis=-1)
        values_2 = tf.squeeze(features_2, axis=-1)

        return out_feat, 0.5*(values+values_2)

    def _make_predictions(self,states,actions,mask=None, test=False,init_state=None,map_state=None):
        features,curr_frame = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)
        # ensembles = features.get_shape()[1]
        # act_feature = tf.expand_dims(self.action_layer[0](actions),1)
        # features =  tf.concat([features, tf.repeat(act_feature,ensembles,axis=1)], axis=2)

        for layer in self.prediction_layers:
            features = layer(features)
        # features = self.predict_layer(features)
        traj = tf.reshape(features,[-1,self.traj_nums,self.traj_length,2])
        return traj,curr_frame


class Represent_Learner(tf.keras.Model):
    def __init__(self,params,state_shape,action_dim,qf,qf_target,dim_shape,hidden_activation="relu", name='Represent_Learner',
    use_sep_tar_encoder=False,use_q_head=False):
        super().__init__(name=name)
        self.params = params
        self.recurrent_layer = layers.MultiHeadAttention(num_heads=params['head_num'], key_dim=128//params['head_num'],output_shape=dim_shape)
        self.transition_layers=tf.keras.Sequential([
            layers.Dense(256,activation=hidden_activation,name='transition_0'),
            layers.Dense(dim_shape,activation=None,name='transition_1')
        ])
        self.action_layer = layers.Dense(64,activation=hidden_activation,name='action_0')

        if use_q_head:
            self.projection_layers=[qf.critic_layers[0]]
            self.projection_layers_target=[qf_target.critic_layers[0]]
        else:
            self.projection_layers=[
                layers.Dense(256, activation=hidden_activation,name='proj_1'),
                layers.Dense(dim_shape, activation=hidden_activation,name='proj_2')
            ]

            self.projection_layers_target=[
                layers.Dense(256, activation=hidden_activation,name='t_proj_1'),
                layers.Dense(dim_shape, activation=hidden_activation,name='t_proj_2')
            ]
        self.use_q_head = use_q_head

        if not use_q_head:
            update_target_variables(self.projection_layers_target.weights,
            self.projection_layers.weights,tau=1.0)

        self.encoder = qf.encoder

        if use_sep_tar_encoder:
            if params['make_prediction']:
                head_num=1#params['head_num']
            else:
                head_num=params['head_num']
            self.target_encoder = RLEncoder(state_shape, action_dim,units=[256]*3,hidden_activation=hidden_activation,
                        state_input=params['state_input'],lstm=params['LSTM'],trans=False,
                        cnn_lstm=params['cnn_lstm'],bptt=params['bptt'],ego_surr=params['ego_surr'],
                        use_trans=params['use_trans'],neighbours=params['neighbours'],time_step=params['time_step'],debug=False,
                        make_rotation=params['make_rotation'],make_prediction=params['make_prediction'],use_map=params['use_map']
                        ,num_traj=params['traj_nums'],path_length=params['path_length'],head_dim=head_num,cnn=params['cnn'],
                        use_hier=params['use_hier'],random_aug=params['random_aug'])
        else:
            self.target_encoder = qf_target.encoder
        

        self.pred_layer = layers.Dense(128,name='prediction_head')

        self.pred_step = params['future_step']

        self.similarity_loss = tf.keras.losses.cosine_similarity

        dummy_state = tf.constant(np.zeros(shape=(1,self.pred_step,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=(1,self.pred_step,) + (action_dim,), dtype=np.float32))
        mask = tf.ones([1,self.pred_step,dummy_state.get_shape()[1]])

        if not self.params['bptt']:
            m=mask
            init_state=None
        else:
            m = mask
            init_state = tf.zeros((1,256))

        if params['use_map'] or params['use_hier']:
            map_s = tf.constant(np.zeros(shape=(1,self.pred_step,) + (state_shape[0]*2,params['path_length'],5), dtype=np.float32))
            self(dummy_state[:,0],map_s[:,0],dummy_action,dummy_state,map_s)
        else:
            map_s = None
            # print(dummy_action)
            self(dummy_state[:,0],None,dummy_action,dummy_state,map_s)
        
        
        self.summary()
    
    def _timestep_attention(self,states,training,mask):
        #[batch,timesteps,dim]
        mask = tf.cast(mask, tf.int16)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        length = mask.get_shape()[-1]
        a = tf.cast(tf.expand_dims(tf.linalg.band_part(tf.ones((length,length)),-1,0),axis=0),tf.int16)
        mask = tf.multiply(a,mask)
        state_val = self.recurrent_layer(states,states,attention_mask=mask,training=True)
        # state_val = tf.nn.relu(state_val)
        return state_val
        
    def _action_transition(self,feat,actions,init_state=None,re=False):
        if self.params['use_map'] or self.params['use_hier']:
            ensembles = feat.get_shape()[1]
            act_feature = tf.expand_dims(self.action_layer(actions),1)
            # act_feature = tf.expand_dims(actions,1)

            features =  tf.concat([feat, tf.repeat(act_feature,ensembles,axis=1)], axis=2)
        else:
            act_feature = self.action_layer(actions)
            features =  tf.concat([feat, act_feature], axis=-1)

        if re:
            if init_state is None:
                init_state = tf.zeros_like(feat)
            return features

        for layer in self.transition_layers:
            features = layer(features)
        return features
    
    def _make_recurrent_rep(self,states,map_state,actions,next_states,next_map_state,mask=None, test=False,init_state=None):
        f_z,f_z_f = [],[]
        # print(next_states.get_shape(),self.pred_step)
        for i in range(self.pred_step):
            if i==0:
                f,_ = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)
            else:
                f,_ = self.encoder(next_states[:,i-1],mask=mask, test=test,init_state=init_state,map_state=next_map_state[:,i-1])
            f_z.append(f)
            if self.params['random_aug']:
                f_n,_ = self.encoder(next_states[:,i],mask=mask, test=test,init_state=init_state,map_state=next_map_state[:,i],
            curr_frames=None)
            else:
                f_n,_ = self.target_encoder(next_states[:,i],mask=mask, test=test,init_state=init_state,map_state=next_map_state[:,i],
                curr_frames=None)
            f_z_f.append(f_n)

        f_z,f_z_f = tf.stack(f_z,1) , tf.stop_gradient(tf.stack(f_z_f,1))
        step_mask = tf.cast(tf.not_equal(next_states[:,:,0,0,0], 0),tf.float32)
        # print(step_mask.get_shape())
        
        f_z = f_z[:,:,0,:]
        f_z_f = f_z_f[:,:,0,:]
        f_a = self.action_layer(actions)
        p_f_z = tf.concat([f_z,f_a], axis=-1)
        z = self._timestep_attention(p_f_z,training=True,mask=step_mask)
        z = self.transition_layers(z)

        p_f_z,f_z_f = self._projection(z,f_z_f)

        p_f_z,f_z_f = self.pred_layer(p_f_z),tf.stop_gradient(f_z_f)

        step_mask = tf.cast(step_mask,tf.float32)

        loss = tf.reduce_mean(tf.multiply(step_mask,self.similarity_loss(f_z_f,p_f_z,axis=-1)))
        return loss

    def _projection(self,feat,feat_target):
        for layer, target_layer in zip(self.projection_layers,self.projection_layers_target):
            if self.params['random_aug']:
                feat,feat_target = layer(feat), layer(feat_target)
            else:
                feat,feat_target = layer(feat), target_layer(feat_target)
        return feat,feat_target
        
    def call(self,states,map_state,actions,next_states,next_map_state,mask=None, test=False,init_state=None):
        # print(map_state)
        return self._make_recurrent_rep(states,map_state,actions,next_states,next_map_state)
        loss = 0
        if map_state is None:
            f_z,curr_frames = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=None)
            f_z = f_z[:,:,:128]
            
            for i in range(self.pred_step):
                
                target_features,_ = self.target_encoder(next_states[:,i],mask=mask, test=test,init_state=init_state,map_state=None,
                curr_frames=None)
                target_features = target_features[:,:,:128]
                target_features = tf.stop_gradient(target_features)

                f_z = self._action_transition(f_z,actions[:,i])

                p_z,p_z_target = self._projection(f_z, target_features)
                p_z_pred = p_z#self.pred_layer(p_z)
                p_z_target = tf.stop_gradient(p_z_target)

                loss += tf.reduce_mean(self.similarity_loss(p_z_target,p_z_pred,axis=-1))/self.pred_step
            
            return loss
        else:
            f_z,curr_frames = self.encoder(states,mask=mask, test=test,init_state=init_state,map_state=map_state)
            for i in range(self.pred_step):
                step_mask = tf.not_equal(next_states[:,i,0,0,0], 0)
                step_mask = tf.expand_dims(tf.cast(step_mask,tf.float32),axis=-1)
                if self.params['random_aug']:
                    target_features,_ = self.encoder(next_states[:,i],mask=mask, test=test,init_state=init_state,map_state=next_map_state[:,i],
                curr_frames=None)
                else:
                    target_features,_ = self.target_encoder(next_states[:,i],mask=mask, test=test,init_state=init_state,map_state=next_map_state[:,i],
                    curr_frames=None)

                # target_features = target_features[:,:,:128]
                target_features = tf.stop_gradient(target_features)

                f_z = self._action_transition(f_z,actions[:,i])
                p_z,p_z_target = self._projection(f_z, target_features)
                p_z_pred = self.pred_layer(p_z)
                p_z_target = tf.stop_gradient(p_z_target)

                loss += tf.reduce_mean(tf.multiply(step_mask,self.similarity_loss(p_z_target,p_z_pred,axis=-1)))/self.pred_step

            return loss
    
    def _update_params(self,tau):
        if self.params['random_aug']:
            return
        update_target_variables(self.projection_layers_target.weights,self.projection_layers.weights,tau)

class SAC_Actor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self,params,state_shape,action_dim, max_action,hidden_activation="relu", name='gaussian_policy'
        ,state_independent_std=False,squash=True,drop_rate=0.1,ensembles=False):
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self._squash = squash
        self.use_map=params['use_map'] or params['use_hier']
        self.params = params
        self.ensemble = ensembles


        self.multi_selection = 'multi_selection' in list(params.keys())
        self._max_action = max_action
        if 'ensembles' not in params:
            self.out_mean = layers.Dense(action_dim, name="L_mean")
            if self._state_independent_std:
                self.out_logstd = tf.Variable(
                    initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                    dtype=tf.float32, name="L_logstd")
            else:
                self.out_logstd = layers.Dense(action_dim, name="L_logstd")
            self.actor_layers = [
                                    layers.Dense(128, activation=hidden_activation),
                                    layers.Dense(32, activation=hidden_activation)]
        else:
            print('en-actor')
            self.out_mean=[]
            self.out_logstd=[]
            self.actor_layers=[]
            for i in range(params['traj_nums']):
                self.out_mean.append(layers.Dense(action_dim, name=f"L_mean_{i}"))
                self.out_logstd.append(layers.Dense(action_dim, name=f"L_logstd_{i}"))
                self.actor_layers.append([
                                        layers.Dense(128, activation=hidden_activation,name=f"a_0_{i}"),
                                        layers.Dense(32, activation=hidden_activation,name=f"a_1_{i}")])
        
        dummy_state = tf.constant(np.zeros(shape=state_shape, dtype=np.float32))
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

        self(dummy_state)
        # if self.make_prediction:
        #     self._make_predictions(dummy_state,mask=m,init_state=init_state,map_state=map_s)
        self.summary()
    
    def call(self,state,test=False):
        features = tf.stop_gradient(state)
        # print(features.get_shape())
        if 'ensembles' in self.params:
            dist = []
            for i in range(self.params['traj_nums']):
                feat = features[:,i]
                for layer in self.actor_layers[i]:
                    feat = layer(feat)
                mean = self.out_mean[i](feat)
                log_std = self.out_logstd[i](feat)
                log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)
                std = tf.exp(log_std)
                dist.append(tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std))
        else:
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
            
            if self.multi_selection or ('multi_actions' in self.params):
                dist =  [tfp.distributions.MultivariateNormalDiag(loc=mean[:,j,:], scale_diag=std[:,j,:]) for j in range(self.traj_nums)]
            else:
                if self.use_map:
                    
                    if self.ensemble:
                        mixture_mean = tf.reduce_mean(mean, axis=1)
                        mixture_var  = tf.abs(tf.reduce_mean(tf.square(std) + tf.square(mean), axis=1) - tf.square(mixture_mean))

                        mean = mixture_mean
                        std = tf.sqrt(mixture_var)
                    
                    else:
                        mean = tf.reduce_mean(mean, axis=1)
                        std = tf.reduce_mean(std, axis=1)
                
                dist =  tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)

        if self.multi_selection or ('multi_actions' in self.params) or ('ensembles' in self.params):
            if test:
                raw_actions = [d.mean() for d in dist]
            else:
                raw_actions = [d.sample() for d in dist]
            log_pis = [d.log_prob(raw_actions[i]) for i,d in enumerate(dist)]
            entropy = [d.entropy() for d in dist]
            raw_actions = tf.stack(raw_actions,axis=1)
            log_pis = tf.stack(log_pis,axis=1)
            entropy = tf.stack(entropy,axis=1)
        else:
            if test:              
                raw_actions = dist.mean()
            else:
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

        return actions,log_pis,entropy
