import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.util import init, check
from ..utils.cnn import CNNBase
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.mix import MIXBase
from ..utils.rnn import RNNLayer
from ..utils.act import ACTLayer
from ..utils.popart import PopArt
from ..util import get_shape_from_obs_space
from ..utils.invariant import Invariant

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal 
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks 
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N 
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart 
        self._recurrent_N = args.recurrent_N 
        self.attn = args.attn
        self.distance = args.distance
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        print('Actor obs shape:', obs_shape)
        self.type_idx_a = [0, 1, 2, 3] + [obs_shape-4, obs_shape-3, obs_shape-2, obs_shape-1]
        # self.type_idx_a = [0, 1, 2, 3] + [obs_shape-5, obs_shape-4, obs_shape-3, obs_shape-2]
        # self.type_idx_a = list(range(8))
        self.type_idx_b = [2, 3, 0, 1, 6, 7, 4, 5]
        
        self._mixed_obs = False
        hidden_dim = 64
        dim_head = 32
        mlp_dim = 32
        if self.attn != 'type_direct_attn':
            self.neighbor_dim = len(self.type_idx_a)
        else:
            self.neighbor_dim = len(self.type_idx_a)*4
            if args.distance != 0:
                self.neighbor_dim += 4
        
        if args.agg == 1 and args.attn != 'direct_concat' and args.attn != 'direct_type_concat':
            self.agg = Invariant(obs_shape, invariant_type=args.attn, hidden_dim=hidden_dim, dim_head=dim_head, mlp_dim=mlp_dim, distance=args.distance, neighbor_dim=self.neighbor_dim)
        self._agg = args.agg
        if self._agg:
            if self.attn == 'direct_concat' or self.attn == 'direct_type_concat':
                obs_shape += len(self.type_idx_a) * 4
                if self.distance != 0:
                     obs_shape += 4       ### 还有距离因素
            else:
                obs_shape = hidden_dim * 2
        print('obs Shape: ', obs_shape)
        self.base = MLPBase(args, obs_shape, use_attn_internal=args.use_attn_internal, use_cat_self=True)
    
        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        self.act = ACTLayer(action_space, input_size, self._use_orthogonal, self._gain)

        if self._use_policy_vhead:
            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
            def init_(m): 
                return init(m, init_method, lambda x: nn.init.constant_(x, 0))
            if self._use_popart:
                self.v_out = init_(PopArt(input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, neighbor_obs=None, neighbor_mask=None, neighbor_type=None, neighbor_relation=None, neighbor_distance=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._agg:
            other_obs = [check(neighbor_obs[:, i]).to(**self.tpdv)[:, self.type_idx_a] for i in range(neighbor_obs.shape[-2])]
            masks = [neighbor_mask[:, i] for i in range(neighbor_mask.shape[-1])]
            other_types = [neighbor_type[:, i] for i in range(neighbor_type.shape[-1])]
            other_relations = [check(neighbor_relation[:, i]).to(**self.tpdv) for i in range(neighbor_relation.shape[-2])]
            other_distances = [check(neighbor_distance[:, i]).to(**self.tpdv) for i in range(neighbor_distance.shape[-2])]   
            if self.attn == 'direct_concat':
                for i in range(neighbor_relation.shape[-2]):
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs_n = torch.cat(other_obs, dim=-1)
            elif self.attn == 'direct_type_concat':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_obs[i][~idxs]
                    other_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs_n = torch.cat(other_obs, dim=-1)
            elif self.attn == 'type_direct_attn':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_obs[i][~idxs]
                    other_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            else:
                obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            
            obs = torch.cat([obs, obs_n], dim=-1)           ### obs也经过了self attention

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, neighbor_obs=None, neighbor_mask=None, neighbor_type=None, neighbor_relation=None, neighbor_distance=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        if self._agg:
            other_obs = [check(neighbor_obs[:, i]).to(**self.tpdv)[:, self.type_idx_a] for i in range(neighbor_obs.shape[-2])]
            masks = [neighbor_mask[:, i] for i in range(neighbor_mask.shape[-1])]
            other_types = [neighbor_type[:, i] for i in range(neighbor_type.shape[-1])]
            other_relations = [check(neighbor_relation[:, i]).to(**self.tpdv) for i in range(neighbor_relation.shape[-2])]
            other_distances = [check(neighbor_distance[:, i]).to(**self.tpdv) for i in range(neighbor_distance.shape[-2])]      ### 之前的distance有bug
            if self.attn == 'direct_concat':
                for i in range(neighbor_relation.shape[-2]):
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs_n = torch.cat(other_obs, dim=-1)
            elif self.attn == 'direct_type_concat':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_obs[i][~idxs]
                    other_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs_n = torch.cat(other_obs, dim=-1)
            elif self.attn == 'type_direct_attn':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_obs[i][~idxs]
                    other_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_obs[i] = torch.cat([other_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_obs[i][masks[i]==0] = -1
                obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            else:
                obs, obs_n = self.agg(obs, other_obs, masks, other_types, other_relations, other_distances)
            
            obs = torch.cat([obs, obs_n], dim=-1)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions, active_masks = active_masks if self._use_policy_active_masks else None)

        values = self.v_out(actor_features) if self._use_policy_vhead else None
       
        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):        
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        
        values = self.v_out(actor_features)

        return values

class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal  
        self._activation_id = args.activation_id     
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.one_hot = args.one_hot
        self.attn = args.attn
        self.distance = args.distance
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]        
        share_obs_shape = get_shape_from_obs_space(share_obs_space)
        
        self.type_idx_a = [0, 1, 2, 3] + [share_obs_shape-4, share_obs_shape-3, share_obs_shape-2, share_obs_shape-1]
        # self.type_idx_a = [0, 1, 2, 3] + [share_obs_shape-5, share_obs_shape-4, share_obs_shape-3, share_obs_shape-2]
        # self.type_idx_a = list(range(8))
        self.type_idx_b = [2, 3, 0, 1, 6, 7, 4, 5]
        self._mixed_obs = False
        # self.onehot_base = nn.Linear(args.num_agents, 64)
        self.onehot_base = nn.Linear(885, 64)

        hidden_dim = 64
        dim_head = 32
        mlp_dim = 32
        if self.attn != 'type_direct_attn':
            self.neighbor_dim = len(self.type_idx_a)
        else:
            self.neighbor_dim = len(self.type_idx_a)*4
            if args.distance != 0:
                self.neighbor_dim += 4

        if args.agg == 1 and args.attn != 'direct_concat' and args.attn != 'direct_type_concat':
            self.agg = Invariant(share_obs_shape, invariant_type=args.attn, hidden_dim=hidden_dim, dim_head=dim_head, mlp_dim=mlp_dim, distance=args.distance, neighbor_dim=self.neighbor_dim)
        self._agg = args.agg
    
        if self.one_hot:
            share_obs_shape += 64

        if self._agg:
            if self.attn == 'direct_concat' or self.attn == 'direct_type_concat':
                share_obs_shape += len(self.type_idx_a) * 4 
                if self.distance != 0:
                    share_obs_shape +=  4
            else:
                share_obs_shape = hidden_dim * 2

        self.base = MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)
        # CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params) if len(share_obs_shape)==3 else MLPBase(args, share_obs_shape, use_attn_internal=True, use_cat_self=args.use_cat_self)

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(share_obs_shape[0], self.hidden_size,
                              self._influence_layer_N, self._use_orthogonal, self._activation_id)
            input_size += self.hidden_size

        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, 1))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, index_one_hot_states=None, neighbor_obs=None, neighbor_mask=None, neighbor_type=None, neighbor_relation=None, neighbor_distance=None):
        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
                if self.one_hot:
                    index_one_hot_states[key] = check(index_one_hot_states[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
            if self.one_hot:
                index_one_hot_states = check(index_one_hot_states).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._agg:
            other_share_obs = [check(neighbor_obs[:, i]).to(**self.tpdv)[:, self.type_idx_a] for i in range(neighbor_obs.shape[-2])]
            masks = [neighbor_mask[:, i] for i in range(neighbor_mask.shape[-1])]
            other_types = [neighbor_type[:, i] for i in range(neighbor_type.shape[-1])]
            other_relations = [check(neighbor_relation[:, i]).to(**self.tpdv) for i in range(neighbor_relation.shape[-2])]
            other_distances = [check(neighbor_distance[:, i]).to(**self.tpdv) for i in range(neighbor_distance.shape[-2])]      ### 之前的distance有bug
            if self.attn == 'direct_concat':
                for i in range(neighbor_relation.shape[-2]):
                    if self.distance != 0:
                        other_share_obs[i] = torch.cat([other_share_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_share_obs[i][masks[i]==0] = -1
                share_obs_n = torch.cat(other_share_obs, dim=-1)
            elif self.attn == 'direct_type_concat':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_share_obs[i][~idxs]
                    other_share_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_share_obs[i] = torch.cat([other_share_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_share_obs[i][masks[i]==0] = -1
                share_obs_n = torch.cat(other_share_obs, dim=-1)
            elif self.attn == 'type_direct_attn':
                for i in range(neighbor_relation.shape[-2]):
                    idxs = np.all(neighbor_relation[:, i]==[1, 0], axis=1)
                    tmp = other_share_obs[i][~idxs]
                    other_share_obs[i][~idxs, :] = tmp[:, self.type_idx_b]
                    if self.distance != 0:
                        other_share_obs[i] = torch.cat([other_share_obs[i], other_distances[i][:, [0]]], dim=-1)
                    other_share_obs[i][masks[i]==0] = -1
                share_obs, share_obs_n = self.agg(share_obs, other_share_obs, masks, other_types, other_relations, other_distances)
            else:
                share_obs, share_obs_n = self.agg(share_obs, other_share_obs, masks, other_types, other_relations, other_distances)
            
            share_obs = torch.cat([share_obs, share_obs_n], dim=-1)

        # if self.one_hot:
        #     share_obs = torch.cat([share_obs, self.onehot_base(index_one_hot_states)], dim=1)
        
        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        values = self.v_out(critic_features)

        return values, rnn_states