import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.target_update_ops import update_target_variables

from .modules.policy import GuassianActor, ValueNet, QNet, Rotater
from .modules.actor_critic_policy import SAC_Actor, SAC_Critic, Represent_Learner


class SAC(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="SAC",
            max_action=1.0,
            lr=3e-4,
            tau=5e-3,
            alpha=0.2,
            auto_alpha=False,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            make_prediction=False,
            pred_trajs=0,
            actor_critic=False,
            actor_critic_target=None,
            bptt=False,
            params=None,
            hidden_activation='relu',
            multi_selection=False,
            multi_actions=False,
            representations=False,
            ensembles=False,
            aug=False,
            sep_rep_opt=False,
            **kwargs):
        super().__init__(
            name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        self.make_prediction = make_prediction
        if self.make_prediction:
            assert pred_trajs>0
            self.pred_loss_fn = Para_MTPLoss(num_modes=pred_trajs)
            self.fut_rotater = Rotater(mode='fut')

        self.actor_critic = actor_critic
        self.actor_critic_target = actor_critic_target
        self.bptt=bptt
        self.hidden_activation = hidden_activation
        self.q_selection  = multi_selection
        self.multi_actions = multi_actions
        self.representations = representations
        self.ensembles = ensembles
        self.aug = aug
        self.sep_rep_opt = sep_rep_opt
        
        if self.representations:
            assert self.actor_critic==True

        self.params = params
        if not self.actor_critic:
            
            self._setup_actor(params,state_shape, action_dim, lr, max_action)
            self._setup_critic_v(params,state_shape,action_dim ,lr)
            self._setup_critic_q(params,state_shape, action_dim, lr)
        else:
            # assert self.actor_critic_target is not None
            print('ac is setting')
            print(state_shape)
            self._setup_actor_critic(params,state_shape, action_dim, lr, max_action)

        # Set hyper-parameters
        self.tau = tau
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32)
            # if self.ensembles:
            #     self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32) #for _ in range(self.params['traj_nums'])]
            #     # print(self.log_alpha)

            self.alpha = tfp.util.DeferredTensor(pretransformed_input=self.log_alpha, transform_fn=tf.exp, dtype=tf.float32)
            #for i in range(self.params['traj_nums'])]
            # print(self.alpha)

            self.target_alpha = -action_dim
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5) 
        else:
            self.alpha = alpha

        self.state_ndim = len(state_shape)

    def _setup_actor_critic(self,params,state_shape, action_dim, lr, max_action=1.):
        self.qf = SAC_Critic(params, state_shape, action_dim)
        self.qf_target = SAC_Critic(params, state_shape, action_dim)

        self.critic_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)
        # if self.ensembles:
        #     self.critic_optimizer = [tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5) for _ in range(self.params['traj_nums'])]
        update_target_variables(self.qf_target.weights, self.qf.weights, tau=1.0)
        hidden_shape = self.qf.h_size
        print(hidden_shape)

        if self.representations:
            # print(self.qf.encoder,self.qf_target.encoder)
            self.rep_func = Represent_Learner(params, state_shape, action_dim, self.qf, self.qf_target,hidden_shape[-1])
            self.rep_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)
        if self.make_prediction:
            self.pred_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)

        self.actor = SAC_Actor(params, hidden_shape, action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)
        # if self.ensembles:
        #     self.actor_optimizer = [tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5) for _ in range(self.params['traj_nums'])]

    def _setup_actor(self, params,state_shape, action_dim, lr, max_action=1.):
        self.actor = GuassianActor(params,state_shape, action_dim, max_action, squash=True,
        hidden_activation=self.hidden_activation)
        self.actor_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, clipnorm=5)

    def _setup_critic_q(self,params, state_shape, action_dim, lr):
        self.qf1 = QNet(params,state_shape, action_dim, name="qf1",hidden_activation=self.hidden_activation)
        self.qf2 = QNet(params,state_shape, action_dim, name="qf2",hidden_activation=self.hidden_activation)
        self.qf1_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)
        self.qf2_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, clipnorm=5)

    def _setup_critic_v(self, params,state_shape,action_dim,lr):
        self.vf = ValueNet(params, state_shape,action_dim)
        self.vf_target = ValueNet(params, state_shape,action_dim)
        update_target_variables(self.vf_target.weights, self.vf.weights, tau=1.0)
        self.vf_optimizer = tf.keras.optimizers.Nadam(learning_rate=lr,clipnorm=5)
    
    def get_hidden_and_val(self,state,mask=None,init_state=None,map_state=None,test=True):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim
        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        h,val = self._get_encoder_val( state,mask,init_state,map_state,test)
        return h.numpy()[0],val.numpy()[0]

    def get_action(self, state,mask=None,init_state=None,map_state=None,test=False):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim
        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        
        if init_state is None:
            action,*res = self._get_action_body(tf.constant(state),mask,init_state,map_state,test)
            if self.q_selection:
                return action.numpy()[0] if is_single_state else action,res[0].numpy()[0]
            return action.numpy()[0] if is_single_state else action
        else:
            action,_,_,h = self._get_action_body(tf.constant(state),mask,init_state,map_state,test)
            return action.numpy()[0] if is_single_state else action , h.numpy()
    
    def get_pca_val(self,state,action,mask=None,init_state=None,map_state=None,test=True):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim
        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state
        action = np.expand_dims(action, axis=0).astype(np.float32)
        h,q = self._pca_val(state,action,mask,init_state,map_state,test)
        return h.numpy()[0],q.numpy()[0]
    
    @tf.function
    def _get_encoder_val(self, state,mask,init_state,map_state,test):
        h,val = self.qf._get_hidden_and_val(state,mask,test,init_state,map_state)
        return h,val
    
    @tf.function
    def _pca_val(self,state,action,mask,init_state,map_state,test):
        feat,q = self.qf._get_pca_val(state,action,mask,test,init_state,map_state)
        return feat,q
    
    @tf.function
    def _multi_selection(self,sample_actions,states,mask,hidden,trajs,map_state):
        q_list = []
        min_list = []
        for i in range(sample_actions.get_shape()[1]):
            if self.multi_actions:
                curr_q1 = self.qf1(states, sample_actions[:,i],mask=mask,init_state=hidden,map_state=map_state)
                curr_q2 = self.qf2(states, sample_actions[:,i],mask=mask,init_state=hidden,map_state=map_state)
            else:
                curr_q1 = self.qf1(states, sample_actions[:,i],mask=mask,init_state=hidden,map_state=map_state,traj=trajs[:,i])
                curr_q2 = self.qf2(states, sample_actions[:,i],mask=mask,init_state=hidden,map_state=map_state,traj=trajs[:,i])
            q_list.append(0.5*(curr_q1+curr_q2))
            # min_list.append())
        q_list = tf.stack(q_list,axis=1)
        ind = tf.argmax(q_list,axis=-1)

        return ind
        
    @tf.function
    def _get_action_body(self, state,mask,init_state,map_state,test):
        if self.actor_critic:
            current_h = self.qf._get_hidden(state,mask=mask,init_state=init_state,map_state=map_state,test=test)
            if self.ensembles:
                sample_actions,*_  = self.actor(current_h)
                # print(sample_actions.get_shape())
                ind = self._ensemble_selections(sample_actions,state,mask,init_state,map_state)
                # tf.print(ind)
                action = sample_actions[:,ind[0]]
                return action,ind,None
            return self.actor(current_h,test=test)
        else:
            if self.q_selection:
                sample_actions,*_ = self.actor(state,mask=mask,init_state=init_state,map_state=map_state,test=test)
                trajs,_ = self.actor._make_predictions(state,mask=mask,init_state=init_state,map_state=map_state)
                inds = self._multi_selection(sample_actions,state,mask,init_state,trajs,map_state)
                inds = tf.cast(inds,tf.int32)
                sample_actions = tf.gather_nd(sample_actions, tf.transpose([tf.range(sample_actions.get_shape()[0]),inds]))
                trajs = tf.gather_nd(trajs, tf.transpose([tf.range(trajs.get_shape()[0]),inds]))
                return sample_actions,trajs,None
            elif self.multi_actions:
                sample_actions,*_ = self.actor(state,mask=mask,init_state=init_state,map_state=map_state,test=test)
                trajs=None
                inds = self._multi_selection(sample_actions,state,mask,init_state,trajs,map_state)
                inds = tf.cast(inds,tf.int32)
                sample_actions = tf.gather_nd(sample_actions, tf.transpose([tf.range(sample_actions.get_shape()[0]),inds]))
                return sample_actions,None,None
            return self.actor(state,mask=mask,init_state=init_state,map_state=map_state,test=test)

        # return actions
    @tf.function
    def _ensemble_selections(self,sample_actions,states,mask,hidden,map_state):
        q_list = []
        for i in range(sample_actions.get_shape()[1]):
            curr_q,currr_q2 = self.qf(states, sample_actions[:,i],mask=mask,init_state=hidden,map_state=map_state)
            m,s = self._cal_std(curr_q,currr_q2)
            q_list.append(m+s)
            # q_list.append(tf.reduce_mean(curr_q,-1) + tf.math.reduce_std(curr_q,-1))
        q_list = tf.stack(q_list,axis=1)
        # tf.print(q_list)
        ind = tf.argmax(q_list,axis=-1)
        # print(ind)
        return ind

    @tf.function
    def _test_rotate(self,state,map_state):
        state_rotater = Rotater(mode='traj')
        map_rotater = Rotater(mode='map')
        state_mask = tf.not_equal(state, 0)[:,:,:,0]
        states,curr_frames = state_rotater(state,state_mask)
        map_mask = tf.not_equal(map_state, 0)[:,:,:,0]
        map_state = map_rotater(map_state,map_mask,curr_frames)

        return states,map_state,state_mask,map_mask
    
    def test_rotate(self,states,map_state):
        return self._test_rotate(states,map_state)

    def train(self, states, actions, next_states, rewards, dones, weights=None,mask=None,hidden=None,
        next_mask=None,next_hidden=None,ego=None,ego_mask=None,map_state=None,next_map_state=None,hist_traj=None,
        future_state=None,future_map_state=None,future_action=None):
        if weights is None:
            weights = np.ones_like(rewards)
        
        if self.actor_critic:

            train_func = self._train_body_actor_critic
        else:
            train_func = self._train_body

        if self.actor_critic:
            td_errors, actor_loss, vf_loss, qf_loss, q_value, logp_min, logp_max, logp_mean, entropy_mean,traj,pred_loss = train_func(
                states, actions, next_states, rewards, dones, weights,mask,hidden,next_mask,next_hidden,ego,ego_mask,
                map_state,next_map_state,hist_traj, future_state,future_map_state ,future_action)
        else:
            td_errors, actor_loss, vf_loss, qf_loss, q_value, logp_min, logp_max, logp_mean, entropy_mean,traj,pred_loss = train_func(
                states, actions, next_states, rewards, dones, weights,mask,hidden,next_mask,next_hidden,ego,ego_mask,
                map_state,next_map_state,hist_traj)

        return td_errors,traj,pred_loss

    def train_rep(self,state,map_state,future_state,future_map_state,future_action):
        return self._represent_train(state,map_state,future_state,future_map_state,future_action).numpy()

    @tf.function
    def _represent_train(self,state,map_state,future_state,future_map_state,future_action):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                simi_loss = self.rep_func(state,map_state,future_action,future_state,future_map_state)
            rep_grad = tape.gradient(simi_loss, self.rep_func.trainable_variables)
            self.rep_optimizer.apply_gradients(zip(rep_grad, self.rep_func.trainable_variables))
            del tape         
        return simi_loss

    @tf.function
    def _train_body_actor_critic(self, states, actions, next_states, rewards, dones, weights,
        mask,hidden,next_mask,next_hidden,ego,ego_mask,map_state,next_map_state,hist_traj,
        future_state,future_map_state,future_action):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2

            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:

                current_q,current_q_2 = self.qf(states, actions,mask=mask,init_state=hidden,map_state=map_state)

                current_h = self.qf._get_hidden(states,mask=mask,init_state=hidden,map_state=map_state)
                sample_actions,logp,entropy = self.actor(current_h)
                next_h = self.qf._get_hidden(next_states,mask=next_mask,init_state=next_hidden,map_state=next_map_state)
                next_actions,next_logp,_ = self.actor(next_h)

                target_q,target_q_2 = self.qf_target(next_states, next_actions,mask=next_mask,init_state=next_hidden,map_state=next_map_state)
                target_value =  tf.stop_gradient(tf.minimum(target_q,target_q_2) - self.alpha * next_logp)
                td_q = tf.stop_gradient(rewards + not_dones * self.discount * target_value)

                if self.aug:
                    current_q_a,current_q_2_a = self.qf(states, actions,mask=mask,init_state=hidden,map_state=map_state,aug=False)
                    next_h_a = self.qf._get_hidden(next_states,mask=next_mask,init_state=next_hidden,map_state=next_map_state,aug=False)
                    next_actions_a,next_logp_a,_ = self.actor(next_h_a)

                    target_q_a,target_q_2_a = self.qf_target(next_states, next_actions_a,mask=next_mask,init_state=next_hidden,map_state=next_map_state,aug=False)
                    target_value_a =  tf.stop_gradient(tf.minimum(target_q_a,target_q_2_a) - self.alpha * next_logp_a)
                    td_q_a = tf.stop_gradient(rewards + not_dones * self.discount * target_value_a)
                    td_q = (td_q + td_q_a)/2

                td_loss_q = 0.5*tf.reduce_mean(weights*((td_q - current_q) ** 2)) + 0.5*tf.reduce_mean(weights*((td_q - current_q_2) ** 2))
                
                if self.aug:
                    td_loss_q = td_loss_q + 0.5*tf.reduce_mean(weights*((td_q - current_q_a) ** 2)) + 0.5*tf.reduce_mean(weights*((td_q - current_q_2_a) ** 2))

                if self.representations and not self.sep_rep_opt:
                    simi_loss =  self.rep_func(states,map_state,future_action,future_state,future_map_state)
                    td_loss_q += simi_loss
                else:
                    simi_loss = None

                sample_q,sample_q_2 = self.qf(states, sample_actions,mask=mask,init_state=hidden,map_state=map_state)

                policy_loss = tf.reduce_mean(weights*(self.alpha * logp - tf.minimum(sample_q,sample_q_2)))

                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean((self.alpha * tf.stop_gradient(logp + self.target_alpha)))
            
            if self.representations:
                var_list = self.qf.critic_layers.weights
                var_list.extend(self.qf.critic_layers_2.weights)
                var_list.extend(self.qf.action_layer.weights)
                var_list.extend(self.qf.action_layer_2.weights)

                var_list.extend(self.qf.encoder.weights)

                if not self.sep_rep_opt:
                    var_list.extend(self.rep_func.recurrent_layer.weights)
                    var_list.extend(self.rep_func.transition_layers.weights)
                    var_list.extend(self.rep_func.action_layer.weights)
                    var_list.extend(self.rep_func.projection_layers.weights)
                    var_list.extend(self.rep_func.pred_layer.weights)

                q_grad = tape.gradient(td_loss_q, var_list)
                self.critic_optimizer.apply_gradients(zip(q_grad, var_list))
            else:
                q_grad = tape.gradient(td_loss_q, self.qf.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(q_grad, self.qf.trainable_variables))

            if self.representations:
                update_target_variables(self.qf_target.weights, self.qf.weights, self.tau)
                self.rep_func._update_params(tau=self.tau)
            else:
                update_target_variables(self.qf_target.weights, self.qf.weights, self.tau)

            # Actor loss
            actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Alpha loss
            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
          
            del tape

        pred_traj = None
        pred_loss = None
        td_loss_v = None
        td_errors = td_loss_q

        return td_errors, policy_loss, td_loss_v, td_loss_q, tf.reduce_mean(current_q), tf.reduce_min(logp), \
                tf.reduce_max(logp), tf.reduce_mean(logp), tf.reduce_mean(entropy),pred_traj,simi_loss
    
    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights,
        mask,hidden,next_mask,next_hidden,ego,ego_mask,map_state,next_map_state,hist_traj):
        with tf.device(self.device):
            assert len(dones.shape) == 2
            assert len(rewards.shape) == 2

            rewards = tf.squeeze(rewards, axis=1)
            dones = tf.squeeze(dones, axis=1)
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:

                current_q1 = self.qf1(states, actions,mask=mask,init_state=hidden,map_state=map_state)
                current_q2 = self.qf2(states, actions,mask=mask,init_state=hidden,map_state=map_state)

                next_v_target = self.vf_target(next_states,mask=next_mask,init_state=next_hidden,map_state=next_map_state)

                target_q = tf.stop_gradient(rewards + not_dones * self.discount * next_v_target)

                td_loss_q1 = tf.reduce_mean(weights*((target_q - current_q1) ** 2))
                td_loss_q2 = tf.reduce_mean(weights*((target_q - current_q2) ** 2)) 

                # Compute loss of critic V
                current_v = self.vf(states,mask=mask,init_state=hidden,map_state=map_state)

                # Resample actions to update V
                sample_actions, logp, entropy,_ = self.actor(states,mask=mask,init_state=hidden,map_state=map_state)  

                current_q1 = self.qf1(states, sample_actions,mask=mask,init_state=hidden,map_state=map_state)
                current_q2 = self.qf2(states, sample_actions,mask=mask,init_state=hidden,map_state=map_state)

                current_min_q = tf.minimum(current_q1, current_q2)

                target_v = tf.stop_gradient(current_min_q - self.alpha * logp)

                td_errors = target_v - current_v
                td_loss_v = tf.reduce_mean(weights*(td_errors ** 2))

                # Compute loss of policy
                policy_loss = tf.reduce_mean(weights*(self.alpha * logp - tf.stop_gradient(current_min_q)))

                pred_traj,pred_loss = None,None
                # print(policy_loss)

                # Compute loss of temperature parameter for entropy
                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean((self.log_alpha * tf.stop_gradient(logp + self.target_alpha)))
                    # print(alpha_loss)

            # Critic Q1 loss
            if self.actor_critic==False:
                q1_grad = tape.gradient(td_loss_q1, self.qf1.trainable_variables)
                self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))

                # Critic Q2 loss
                q2_grad = tape.gradient(td_loss_q2, self.qf2.trainable_variables)
                self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

                # Critic V loss
                vf_grad = tape.gradient(td_loss_v, self.vf.trainable_variables)
                self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))
                # Update Target V
                update_target_variables(self.vf_target.weights, self.vf.weights, self.tau)

                # Actor loss
                actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
            else:
                grad = tape.gradient(loss, self.actor_critic.trainable_weights)
                self.actor_critic_optimizer.apply_gradients(zip(grad,self.actor_critic.trainable_weights))
                update_target_variables(self.actor_critic_target.get_vf_weights(), self.actor_critic.get_vf_weights(), self.tau)

            # Alpha loss
            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
          
            del tape

        return td_errors, policy_loss, td_loss_v, td_loss_q1, tf.reduce_mean(current_min_q), tf.reduce_min(logp), \
                tf.reduce_max(logp), tf.reduce_mean(logp), tf.reduce_mean(entropy),pred_traj,pred_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones,mask=None,hidden=None,
        next_mask=None,next_hidden=None,ego=None,ego_mask=None,map_state=None,next_map_state=None):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)

        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones,mask,hidden,next_mask,next_hidden,
        ego,ego_mask,map_state,next_map_state)

        return td_errors.numpy()

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones,mask,hidden,next_mask,next_hidden,
    ego,ego_mask,map_state,next_map_state):
        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)

            # Compute TD errors for Q-value func
            current_q1 = self.qf1(states, actions,mask=mask,init_state=hidden,map_state=map_state)
            vf_next_target = self.vf_target(next_states,mask=next_mask,init_state=next_hidden,map_state=next_map_state)

            target_q = tf.stop_gradient(rewards + not_dones * self.discount * vf_next_target)

            td_errors_q1 = target_q - current_q1

        return td_errors_q1

    @staticmethod
    def get_argument(parser=None):
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--auto-alpha', action="store_true")

        return parser
