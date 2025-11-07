import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import deque
from copy import deepcopy
import signal
import setproctitle

import numpy as np
import torch
import torch.nn.functional as F
from simulet import Engine, TlPolicy, Verbosity
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ppo.buffer import SharedReplayBuffer
from ppo.ppo import R_MAPPO as TrainAlgo
from ppo.algorithm.Policy import R_MAPPOPolicy as Policy
from ppo.config import get_config
from simulet.tl import TrafficLights,LightState
from simulet import Engine, TlPolicy, Verbosity, LaneChange
from simulet.map import LaneType, LaneTurn

torch.set_float32_matmul_precision('medium')

def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out

def _t2n(x):
    return x.detach().cpu().numpy()

class Env:
    def __init__(self, data_path, step_size, step_count, log_dir, base_algo, reward, interval, alpha=0, agent_file=None, yellow_time=0, save=False):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir+'/details'):  
            os.makedirs(self.log_dir+'/details', exist_ok=True)
        junction_blocking_count = 5
        if agent_file is None:
            agent_file = 'agents.bin'
        if 'cleaned' in data_path:
            buffer_size = 1300
        else:
            buffer_size = 600
        self.eng = Engine(
            map_file=f'{data_path}/map.bin',
            agent_file = f'{data_path}/'+agent_file,
            tl_file=f'{data_path}/tls.bin',
            id_file='',
            start_step=0,
            total_step=1 << 30,
            step_interval=1,
            seed=43,
            verbose_level=Verbosity.NO_OUTPUT,
            agent_limit=-1,
            disable_aoi_out_control=False,
            junction_blocking_count = junction_blocking_count,
            junction_yellow_time=yellow_time,
            lane_change=LaneChange.MOBIL,
            lane_veh_add_buffer_size=buffer_size,
            lane_veh_remove_buffer_size=buffer_size
        )
        print('Paralleled engines created!')
        self.alpha = alpha
        M = self.eng.get_map()
        TL=TrafficLights(f'{data_path}/tls.bin')
        self.action_sizes = list(self.eng.get_junction_phase_counts())

        def lanes_collect(js):
            in_lanes_list = []
            out_lanes_list = []
            phase_lanes_list = []
            phase_label_list = []

            if os.path.exists(f'{data_path}/selected_junctions.json'):
                js = json.load(open(f'{data_path}/selected_junctions.json'))
                print('Junction Num:', len(js))

            for jid in js:
                junction = M.junction_map[jid]
                phases_lane = []
                in_lane, out_lane = [], []
                labels = []
                if junction.id in TL.tls_map:
                    tl = TL.tls_map[junction.id]
                    for phase in tl.phases:
                        lanes = [i for i,j in zip(junction.lanes,phase.states) if j==LightState.GREEN and i.type==LaneType.DRIVING and i.turn!=LaneTurn.RIGHT and i.turn!=LaneTurn.AROUND]        
                        in_lanes = [m.predecessors[0].id for m in lanes]
                        out_lanes = [m.successors[0].id for m in lanes]
                        phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                        in_lane+=in_lanes
                        out_lane+=out_lanes
                        labels.append([
                            any(i.turn==LaneTurn.STRAIGHT for i in lanes), 
                            any(i.turn==LaneTurn.LEFT for i in lanes)
                            ])
                in_lanes_list.append(list(set(in_lane)))
                out_lanes_list.append(list(set(out_lane)))
                phase_lanes_list.append(phases_lane)
                phase_label_list.append(labels)
            return in_lanes_list, out_lanes_list, phase_lanes_list, phase_label_list
    
        js = self.eng.get_junction_ids()
        self.in_lanes, self.out_lanes, self.jpl, self.jpl_label = lanes_collect(js)
        self.max_action_size = max(self.action_sizes)

        def in_lane_numpy(in_lanes):
            max_in_lane_num = max([len(i) for i in in_lanes])
            in_lanes_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            in_lanes_zero_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            for i, in_lane in enumerate(in_lanes):
                in_lanes_array[i, :len(in_lane)] = in_lane
                in_lanes_zero_array[i, len(in_lane):] = 1
            return in_lanes_array, in_lanes_zero_array

        def phase_lane_numpy(phase_lanes):
            max_inflow_lane_num = max_outflow_lane_num = max(max([max([len(j[1]) for j in i]) for i in phase_lanes]),  max([max([len(j[0]) for j in i]) for i in phase_lanes]))
            
            phase_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            phase_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            zero_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            zero_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            non_zeros_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            non_zeros_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            missing_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            missing_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            missing_phase = np.zeros((len(phase_lanes), self.max_action_size), dtype=int)

            for i, phase_lane in enumerate(phase_lanes):
                for j, lanes in enumerate(phase_lane):
                    phase_lanes_inflow[i, j, :len(lanes[0])] = lanes[0]
                    phase_lanes_outflow[i, j, :len(lanes[1])] = lanes[1]
                    zero_lanes_inflow[i, j, len(lanes[0]):] = 1
                    zero_lanes_outflow[i, j, len(lanes[1]):] = 1
                    non_zeros_inflow[i, j, :len(lanes[0])] = 1
                    non_zeros_outflow[i, j, :len(lanes[1])] = 1
                if len(phase_lane) < self.max_action_size:
                    for j in range(len(phase_lane), self.max_action_size):
                        missing_lanes_inflow[i, j, :] = 1
                        missing_lanes_outflow[i, j, :] = 1
                        non_zeros_inflow[i, j, :] = 1
                        non_zeros_outflow[i, j, :] = 1
                        missing_phase[i, j] = 1
            return phase_lanes_inflow, phase_lanes_outflow, zero_lanes_inflow, zero_lanes_outflow, missing_lanes_inflow, missing_lanes_outflow, non_zeros_inflow, non_zeros_outflow, missing_phase

        a = []
        if os.path.exists(f'{data_path}/selected_junctions.json'):
            js = json.load(open(f'{data_path}/selected_junctions.json'))
            j_map = {j: i for i, j in enumerate(self.eng.get_junction_ids())}
            self.jids = [j_map[i] for i in js]
            self.action_sizes = [self.action_sizes[i] for i in self.jids]
            self.max_action_size = max(self.action_sizes)
            self.available_actions = np.zeros((len(self.action_sizes), max(self.action_sizes)))
            for i, j in enumerate(self.action_sizes):
                self.available_actions[i, :j] = 1
            self.phase_lanes = self.jpl
            self.junction_type_list = [np.array([(m==3)|(m==6), (m==4)|(m==8)]).astype(int) for m in self.action_sizes]
            self.junction_scale_list = np.array([[len(i)/j] for i, j in zip(self.in_lanes, self.action_sizes)])
        else:
            self.jids = range(len(self.action_sizes))
            self.available_actions = np.zeros((len(self.action_sizes), max(self.action_sizes)))
            for i, j in enumerate(self.action_sizes):
                self.available_actions[i, :j] = 1
            self.phase_lanes = self.jpl
            self.junction_type_list = [np.array([(m==3)|(m==6), (m==4)|(m==8)]).astype(int) for m in self.action_sizes]
            self.junction_scale_list = np.array([[len(i)/j] for i, j in zip(self.in_lanes, self.action_sizes)])
        
        self.phase_lanes_inflow, self.phase_lanes_outflow, self.zero_lanes_inflow, self.zero_lanes_outflow, self.missing_lanes_inflow, self.missing_lanes_outflow, self.non_zeros_inflow, self.non_zeros_outflow, self.missing_phases = phase_lane_numpy(self.phase_lanes)
        self.in_lane_array, self.zero_in_lane_array = in_lane_numpy(self.in_lanes)

        self.junction_phase_sizes = np.zeros((len(self.jids), max(self.action_sizes) + 1))
        for i, phases in enumerate(self.phase_lanes):
            for j, phase in enumerate(phases):
                self.junction_phase_sizes[i, j] = len(phase[0])
            self.junction_phase_sizes[i, -1] = len(phases)

        self.junction_graph_edges, self.junction_graph_edge_feats = [], []
        self.junction_src_list = [[] for _ in range(len(self.jids))]
        for src_idx, out_lanes in tqdm(enumerate(self.out_lanes)):
            for out_lane in out_lanes:
                ids = [jid for (jid, in_lanes) in enumerate(self.in_lanes) if out_lane in in_lanes]
                if len(ids) != 0:
                    if [src_idx, ids[0]] not in self.junction_graph_edges:
                        self.junction_graph_edges.append([src_idx, ids[0]])
                        self.junction_graph_edge_feats.append([M.lanes[out_lane].length, 1])
                        self.junction_src_list[ids[0]].append(src_idx)
                    else:
                        idx = [i for i, value in enumerate(self.junction_graph_edges) if value == [src_idx, ids[0]]][0]
                        self.junction_graph_edge_feats[idx][1] += 1
        self.max_neighbor_num = max([len(i) for i in self.junction_src_list])
        print('Max Neighbor Number:', self.max_neighbor_num)
        self.connect_matrix = np.zeros((len(self.jids), len(self.jids)))
        for i, idxs in enumerate(self.junction_src_list):
            if len(idxs) == 0:       
                continue
            for j in idxs:
                self.connect_matrix[j, i] = 1/len(idxs)

        if base_algo == 'mp_builtin':
            for i in range(self.eng.junction_count):
                self.eng.set_tl_duration_batch(list(range(self.eng.junction_count)), interval)
                self.eng.set_tl_policy_batch(list(range(self.eng.junction_count)), TlPolicy.MAX_PRESSURE)
                self.eng.set_tl_policy_batch(self.jids, TlPolicy.MANUAL)

        self.phase_relation = np.zeros((len(self.jids), self.max_neighbor_num, 2), dtype=int)   ### one-hot type information
        self.neighbor_type = np.zeros((len(self.jids), self.max_neighbor_num), dtype=int)   
        self.neighbor_mask = np.zeros((len(self.jids), self.max_action_size), dtype=int)
        
        for dst_idx in range(len(self.jids)):
            src_idxs = self.junction_src_list[dst_idx] 
            if len(src_idxs) == 0:
                continue
            self.neighbor_mask[:len(src_idxs)] = 1
            dst_phase_lanes = self.phase_lanes[dst_idx]
            dst_in_lanes = self.in_lanes[dst_idx]
            for idx, src_idx in enumerate(src_idxs):
                src_phase_lanes = self.phase_lanes[src_idx]
                src_out_lanes = self.out_lanes[src_idx]
                for dst_phase_idx, dst_phase in enumerate(dst_phase_lanes):
                    if set(dst_phase[0])&set(src_out_lanes) == set():
                        continue
                    if len(src_phase_lanes) > 2:
                        if set(src_phase_lanes[2][1])&set(dst_in_lanes) != set() or set(src_phase_lanes[1][1])&set(dst_in_lanes) != set():
                            self.phase_relation[dst_idx, idx, :] = [1, 0]
                        else:
                            self.phase_relation[dst_idx, idx, :] = [0, 1]
                    elif len(src_phase_lanes) == 2:
                        if set(src_phase_lanes[1][1])&set(dst_in_lanes) != set():
                            self.phase_relation[dst_idx, idx, :] = [1, 0]
                        else:
                            self.phase_relation[dst_idx, idx, :] = [0, 1]
                    if dst_phase_idx in [0, 1]:
                        self.neighbor_type[dst_idx, idx] = 0
                    else:
                        self.neighbor_type[dst_idx, idx] = 1
        
        self.edge_distance = np.zeros((len(self.jids), self.max_neighbor_num, 1))
        self.edge_strength = np.zeros((len(self.jids), self.max_neighbor_num, 1))

        for i, idxs in enumerate(self.junction_src_list):
            if len(idxs) == 0:
                continue
            self.edge_distance[i, :len(idxs), :] = np.array([min(self.junction_graph_edge_feats[self.junction_graph_edges.index([j, i])][0], 500) for j in idxs]).reshape(-1, 1)    # 超过500米的邻居也认为是500米算
            self.edge_strength[i, :len(idxs), :] = np.array([self.junction_graph_edge_feats[self.junction_graph_edges.index([j, i])][1] for j in idxs]).reshape(-1, 1)

        self.edge_distance = self.edge_distance/100
        self.edge_distance = np.concatenate([self.edge_distance, self.edge_strength], axis=2)

        self.junction_src_list_rearraged = [[] for _ in range(len(self.jids))]
        self.phase_relation_rearraged = np.zeros((len(self.jids), self.max_neighbor_num, 2), dtype=int)
        self.neighbor_type_rearraged = np.zeros((len(self.jids), self.max_neighbor_num), dtype=int)
        self.edge_distance_rearraged = np.zeros((len(self.jids), self.max_neighbor_num, 2))
        self.neighbor_mask_rearraged = np.zeros((len(self.jids), self.max_neighbor_num), dtype=int)

        for dst_idx in range(len(self.jids)):
            src_idxs = self.junction_src_list[dst_idx] 
            if len(src_idxs) == 0:
                continue
            src_idxs_new = np.zeros(4, dtype=int)
            idxs_new = np.zeros(4, dtype=int)
            typea_idxs = [i for i, j in enumerate(self.neighbor_type[dst_idx, :len(src_idxs)]) if j == 0]
            src_idxs_new[:len(typea_idxs)] = [src_idxs[i] for i in typea_idxs]
            self.neighbor_mask_rearraged[dst_idx, :len(typea_idxs)] = 1
            idxs_new[:len(typea_idxs)] = typea_idxs
            typeb_idxs = [i for i, j in enumerate(self.neighbor_type[dst_idx, :len(src_idxs)]) if j == 1]
            src_idxs_new[2:2+len(typeb_idxs)] = [src_idxs[i] for i in typeb_idxs]
            self.neighbor_mask_rearraged[dst_idx, 2:2+len(typeb_idxs)] = 1
            idxs_new[2:2+len(typeb_idxs)] = typeb_idxs
            self.junction_src_list_rearraged[dst_idx] = src_idxs_new.tolist()
            self.phase_relation_rearraged[dst_idx, :] = self.phase_relation[dst_idx, idxs_new]
            self.neighbor_type_rearraged[dst_idx, :] = self.neighbor_type[dst_idx, idxs_new]
            self.edge_distance_rearraged[dst_idx, :] = self.edge_distance[dst_idx, idxs_new]

        print(f'Training on {len(self.jids)} junctions')
        self._cid = self.eng.make_checkpoint()
        self.step_size = step_size
        self.step_count = step_count
        self._step = 0
        self.reward = reward
        self.info = {
            'ATT': 1e999,
            'Throughput': 0,
            'reward': 0,
            'ATT_inside': 1e999,
            'ATT_finished': 1e999,
            'Throughput_inside': 0,
        }
        self.data_path = data_path
        self.one_hot_mapping_matrix = np.eye(self.max_action_size)

    def reset(self):
        self.eng.restore_checkpoint(self._cid)
    
    def observe(self):   
        cnt = self.eng.get_lane_waiting_vehicle_counts()
        in_cnt_states, out_cnt_states = cnt[self.phase_lanes_inflow], cnt[self.phase_lanes_outflow]
        in_cnt_states[self.zero_lanes_inflow==1] = 0
        in_cnt_states[self.missing_lanes_inflow==1] = -1
        n = np.sum(in_cnt_states, axis=2)
        n[self.missing_phases==1] = -1
        observe_states = np.concatenate([n, self.junction_phase_sizes], axis=1)
        return observe_states, observe_states

    def step(self, action):
        self.eng.set_tl_phase_batch(self.jids, action)
        self.eng.next_step(self.step_size)
        s, shared_s = self.observe()

        actions_one_hot = self.one_hot_mapping_matrix[np.array(action).reshape(-1), :]
        s = np.concatenate([s, actions_one_hot], axis=1)
        shared_s = np.concatenate([shared_s, np.array(actions_one_hot)], axis=1)

        cnt = self.eng.get_lane_waiting_vehicle_counts()
        if self.reward == 'queue':
            r = cnt[self.in_lane_array]
            r[self.zero_in_lane_array==1] = 0
            r = -np.sum(r, axis=1)
        if self.reward == 'sum_queue':
            r = [-np.mean([np.sum(cnt[phase_lanes[0]]) for phase_lanes in self.phase_lanes[i]]) for i in range(len(self.jids))]
        if self.reward == 'one_hop_queue':            
            r = cnt[self.in_lane_array]
            r[self.zero_in_lane_array==1] = 0
            r = -np.sum(r, axis=1)
            r_neighbour = np.dot(r, self.connect_matrix)
            r = r + self.alpha*r_neighbour
        if self.reward == 'one_hop_sum_queue':
            r = cnt[self.in_lane_array]
            r[self.zero_in_lane_array==1] = 0
            r = -np.sum(r, axis=1)
            r = r + self.alpha*np.dot(r, self.connect_matrix)
        if self.reward == 'pressure':
            r = [-np.abs(np.sum(cnt[self.in_lanes[i]])-np.sum(cnt[self.out_lanes[i]])) for i in range(len(self.in_lanes))]
        if self.reward == 'one_hop_pressure':
            r = np.array([-np.abs(np.sum(cnt[self.in_lanes[i]])-np.sum(cnt[self.out_lanes[i]])) for i in range(len(self.in_lanes))])
            r_neighbour = np.dot(r, self.connect_matrix)
            r = r + self.alpha*r_neighbour
        self.info['reward'] = np.mean(r)

        self._step += 1
        done = False
        if self._step >= self.step_count:
            self.info['ATT'] = self.eng.get_departed_vehicle_average_traveling_time()
            self.info['ATT_finished'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            pickle.dump((
                self.eng.get_vehicle_statuses(), self.eng.get_vehicle_traveling_or_departure_times()
            ), open(self.log_dir+'/details/'+time.strftime('%Y%m%d_%H%M%S.pkl'), 'wb'))
            self.info['VEH'] = self.eng.get_running_vehicle_count()
            self._step = 0
            self.reset()
            done = True
        return s, shared_s, r, done, self.info


def make_mlp(*sizes, act=nn.GELU, dropout=0.1):
    layers = []
    for i, o in zip(sizes, sizes[1:]):
        layers.append(nn.Linear(i, o))
        layers.append(act())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers[:-2])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def lerp(a, b, t):
    t = min(1, t)
    return a*(1-t)+b*t

@torch.no_grad()
def collect(buffer, trainer, step, args):
    actions_env = []
    trainer.prep_rollout()
    if args.agg:
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = trainer.policy.get_actions(np.concatenate(buffer.share_obs[step]),
                            np.concatenate(buffer.obs[step]),
                            np.concatenate(buffer.rnn_states[step]),
                            np.concatenate(buffer.rnn_states_critic[step]),
                            np.concatenate(buffer.masks[step]), 
                            available_actions=np.concatenate(buffer.available_actions[step]), 
                            neighbor_obs=np.concatenate(buffer.neighbor_obs[step]),
                            neighbor_masks=np.concatenate(buffer.neighbor_mask[step]), 
                            neighbor_types=np.concatenate(buffer.neighbor_type_matrix[step]), 
                            neighbor_relations=np.concatenate(buffer.neighbor_relation[step]),
                            neighbor_distances=np.concatenate(buffer.neighbor_distances_matrix[step]))
    else:
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = trainer.policy.get_actions(np.concatenate(buffer.share_obs[step]),
                            np.concatenate(buffer.obs[step]),
                            np.concatenate(buffer.rnn_states[step]),
                            np.concatenate(buffer.rnn_states_critic[step]),
                            np.concatenate(buffer.masks[step]), 
                            available_actions=np.concatenate(buffer.available_actions[step]))
    # [self.envs, agents, dim]
    values = np.expand_dims(_t2n(value), 0)
    actions = np.expand_dims(_t2n(action), 0)
    action_log_probs = np.expand_dims(_t2n(action_log_prob), 0)
    rnn_states = np.expand_dims(_t2n(rnn_states), 0)
    rnn_states_critic = np.expand_dims(_t2n(rnn_states_critic), 0)
    # rearrange action
    
    actions_env = [a for a in actions[0,:,0]]      
    return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

def insert(args, data, buffer):
    obs, shared_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data
    dones = np.ones((args.n_rollout_threads,args.num_agents), dtype=int)*dones
    rnn_states[dones==True] = 0
    rnn_states_critic[dones==True] = 0
    masks = np.ones((args.n_rollout_threads, args.num_agents, 1), dtype=np.float32)
    masks[dones==True] = 0
    obs = np.expand_dims(np.array(obs),axis=0)

    rewards = np.array(rewards).reshape(args.n_rollout_threads,args.num_agents,1)
    buffer.insert(shared_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions, base_idx=0, update=True)

@torch.no_grad()
def compute(buffer, trainer, args):
    trainer.prep_rollout()
    if args.agg:
        next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                np.concatenate(buffer.rnn_states_critic[-1]),
                                                np.concatenate(buffer.masks[-1]), 
                                                neighbor_obs=np.concatenate(buffer.neighbor_obs[-1]),
                                                neighbor_masks=np.concatenate(buffer.neighbor_mask[-1]),
                                                neighbor_types=np.concatenate(buffer.neighbor_type_matrix[-1]), 
                                                neighbor_relations=np.concatenate(buffer.neighbor_relation[-1]),
                                                neighbor_distances=np.concatenate(buffer.neighbor_distances_matrix[-1]))
    else:
        next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                np.concatenate(buffer.rnn_states_critic[-1]),
                                                np.concatenate(buffer.masks[-1]))
    buffer.compute_returns(np.expand_dims(_t2n(next_values), 0), trainer.value_normalizer)

def train(buffer, trainer):
    trainer.prep_training()
    train_infos = trainer.train(buffer)      
    buffer.after_update()
    return train_infos

def save(trainer, save_dir, type='present'):
    policy_actor = trainer.policy.actor
    torch.save(policy_actor.state_dict(), str(save_dir) + "/actor_{}.pt".format(type))
    policy_critic = trainer.policy.critic
    torch.save(policy_critic.state_dict(), str(save_dir) + "/critic_{}.pt".format(type))

def load(trainer, load_dir):
    trainer.policy.actor.load_state_dict(torch.load(str(load_dir) + "/actor_present.pt"))
    trainer.policy.critic.load_state_dict(torch.load(str(load_dir) + "/critic_present.pt"))
    print('Load Pretrained!')

def main():
    parser = get_config()
    parser.add_argument('--data', type=str, default='../data/beijing_chaoyang')
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=15)#3600/30=120 episode length
    parser.add_argument('--algo', choices=['ft_builtin', 'mp_builtin'], default='mp_builtin')
    parser.add_argument('--training_step', type=int, default=1000000)
    parser.add_argument('--buffer_size', type=int, default=10240)
    parser.add_argument('--reward', type=str, default='pressure')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--hete_ppo", type=int, default=0, help='hete_ppo (default: 0)')
    parser.add_argument('--mlp', type=str, default='256,256')
    parser.add_argument('--buffer_episode_size', type=int, default=8)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument("--ppo_epoch", type=int, default=15,help='number of ppo epochs (default: 15)')
    parser.add_argument("--load", type=int, default=0,help='pretrain (default: 0)')
    parser.add_argument("--one_hot", type=int, default=0,help='whether to use mapped one-hot embedding in the critic input')
    parser.add_argument("--alpha", type=float, default=0.2,help='balance of neighbour rewards')
    parser.add_argument("--agg", type=int, default=0, help='whether to use attention to aggregate neighbor information')
    parser.add_argument("--agent_file", type=str, default=None)
    parser.add_argument("--attn", type=str, default='attn_sum', help='attention method')
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--yellow', type=int, default=0, help='yellow time duration')
    parser.add_argument('--distance', type=int, default=0, help='whether to use distance as edge feature')
    parser.add_argument("--num_mini_batch", type=int, default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument('--save', type=int, default=0, help='whether to save the model')
    parser.add_argument("--layer_N", type=int, default=1, help="Number of layers for actor/critic networks")    
    parser.add_argument("--use_multi_step_lr_decay", action='store_true', default=False, help='use a linear schedule on the learning rate')
    
    args = parser.parse_args()

    args.city = args.data.split('/')[-1]

    path = 'log/ppo_simulet2_universal/{}_{}_lr_{}_buffersize_{}_ppoepoch_{}_obs_mean_{}_interval_{}_agg_{}_clip_{}_attn_{}_distance_{}_lrdecay_{}_{}'.format(args.yellow, args.city, args.lr, args.buffer_episode_size, args.ppo_epoch, args.reward, args.interval, args.agg, args.clip_param, args.attn, args.distance, args.use_multi_step_lr_decay, time.strftime('%d%H%M'))
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/cmd.sh', 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\ntensorboard --port 8888 --logdir '+os.path.abspath(path))
    with open(f'{path}/args.json', 'w') as f:
        json.dump(vars(args), f)
    print('tensorboard --port 8888 --logdir '+os.path.abspath(path))

    writer = SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

    if args.algo == 'ft_builtin':
        base_algo = TlPolicy.FIXED_TIME
    elif args.algo == 'mp_builtin':
        base_algo = TlPolicy.MAX_PRESSURE
    else:
        raise NotImplementedError
        

    env = Env(
        data_path=args.data,
        step_size=args.interval,
        step_count=args.steps//args.interval,
        log_dir=path,
        base_algo=base_algo,
        reward=args.reward,
        interval=args.interval,
        alpha=args.alpha,
        agent_file=args.agent_file,
        yellow_time=args.yellow,
        save=args.save
    )
    args.num_agents = len(env.jids)
    args.episode_length = args.steps//args.interval*args.buffer_episode_size
    
    obs, shared_obs = env.observe()
    one_hot_mapping_matrix = np.eye(env.max_action_size)
    actions = np.zeros(args.num_agents, dtype=int)
    actions_one_hot = one_hot_mapping_matrix[actions, :]
    obs = np.concatenate([obs, actions_one_hot], axis=1)
    shared_obs = np.concatenate([shared_obs, actions_one_hot], axis=1)
    obs_sizes = [len(i) for i in obs]
    share_obs_sizes = [len(i) for i in shared_obs]
    max_neighbor_num = env.max_neighbor_num
    print(f'{len(obs_sizes)} agents:')
    policy = Policy(args,
                        obs_sizes[0],
                        share_obs_sizes[0],
                        env.max_action_size,
                        device = device)
    trainer = TrainAlgo(args, policy, device = device)
    if args.load:
        load(trainer, args.data+'/models')
    if args.agg == 1:
        buffer = SharedReplayBuffer(args,
                                    len(obs_sizes),
                                    obs_sizes[0],
                                    share_obs_sizes[0],
                                    env.max_action_size, 
                                    max_neighbor_num=max_neighbor_num,
                                    parallel_num=1,
                                    phase_relation=env.phase_relation_rearraged,
                                    neighbor_type=env.neighbor_type_rearraged,
                                    neighbor_idxs=env.junction_src_list_rearraged, 
                                    neighbor_distances=env.edge_distance_rearraged,
                                    neighbor_masks=env.neighbor_mask_rearraged)
    else:
        buffer = SharedReplayBuffer(args,
                            len(obs_sizes),
                            obs_sizes[0],
                            share_obs_sizes[0],
                            env.max_action_size, 
                            parallel_num=1)
    
    buffer.share_obs[0] = np.expand_dims(np.array(shared_obs),axis=0)
    buffer.obs[0] = np.expand_dims(np.array(obs),axis=0)
    if args.agg == 1:
        buffer.cal_neighbor_obs(buffer.obs[0], 0, update=True)
    buffer.available_actions[0] = np.expand_dims(np.array(env.available_actions),axis=0)
    episode_step = 0
    episode_reward = 0
    episode_count = 0
    episode_num = 0
    available_actions = env.available_actions
    best_episode_reward = -1e999

    def handle_signal(signum, frame):
        save(trainer, path, 'present')
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_signal)

    with tqdm(range(args.training_step), ncols=100, smoothing=0.1) as bar:
        for step in bar:
            _st = time.time()
            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = collect(buffer, trainer, episode_step, args)
            next_obs, next_shared_obs, rewards, dones, infos = env.step(actions_env)
            episode_reward += infos['reward']
            
            data = next_obs, next_shared_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions
            # insert data into buffer
            
            insert(args, data, buffer)

            episode_count += 1
            episode_step += 1
            if episode_step % ((args.steps//args.interval)*args.buffer_episode_size) == 0:   
                compute(buffer, trainer, args)
                train_infos = train(buffer, trainer)
                bar.set_description(f'Veh: {infos["VEH"]}, ATT: {infos["ATT"]:.2f}, TP: {infos["Throughput"]}')
                episode_step = 0

            if dones:
                #add training update
                writer.add_scalar('metric/EpisodeReward', episode_reward / episode_count, episode_num)
                if episode_reward / episode_count > best_episode_reward:
                    best_episode_reward = episode_reward / episode_count
                    writer.add_scalar('metric/Best_EpisodeReward', episode_reward / episode_count)    
                    writer.add_scalar('metric/Best_ATT', infos['ATT'])
                    writer.add_scalar('metric/Best_ATT_finished', infos['ATT_finished'])
                    writer.add_scalar('metric/Best_Throughput', infos['Throughput'])
                    save(trainer, path, 'best')
                episode_num += 1
                episode_reward = 0
                episode_count = 0
                writer.add_scalar('metric/ATT', infos['ATT'], step)
                writer.add_scalar('metric/ATT_finished', infos['ATT_finished'], step)
                writer.add_scalar('metric/Throughput', infos['Throughput'], step)
                writer.add_scalar('metric/VEH', infos['VEH'], step)
            writer.add_scalar('metric/Reward', infos['reward'], step)            
            writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
        writer.close()

        
if __name__ == '__main__':
    main()
