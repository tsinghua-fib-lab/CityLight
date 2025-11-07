import numpy as np
import torch
from .actor_critic import R_Actor, R_Critic
from ..util import update_linear_schedule


class R_MAPPOPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        ### 定義multistep的lr scheduler
        if args.use_multi_step_lr_decay:
            self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=[1000, 1500], gamma=0.4)
            self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=[1000, 1500], gamma=0.4)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False, index_one_hot_states=None, neighbor_obs=None, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, index_one_hot_states, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks, index_one_hot_states=None, neighbor_obs=None, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, index_one_hot_states, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None, index_one_hot_states=None, neighbor_obs=None, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        action_log_probs, dist_entropy, policy_values = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, index_one_hot_states, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        return values, action_log_probs, dist_entropy, policy_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, neighbor_obs=None, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_masks, neighbor_type=neighbor_types, neighbor_relation=neighbor_relations, neighbor_distance=neighbor_distances)
        return actions, rnn_states_actor
    

class R_MAPPOPolicy_HETE:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor_3phase = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic_3phase = R_Critic(args, self.share_obs_space, self.device)

        self.actor_4phase = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic_4phase = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(list(self.actor_3phase.parameters())+list(self.actor_4phase.parameters()), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_3phase.parameters())+list(self.critic_4phase.parameters()), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False, three_phases_idxs=None, four_phases_idxs=None):
        action_3phase, action_log_probs_3phase, rnn_states_actor_3phase = self.actor_3phase(obs[three_phases_idxs], rnn_states_actor[three_phases_idxs], masks[three_phases_idxs], available_actions[three_phases_idxs], deterministic)
        action_4phase, action_log_probs_4phase, rnn_states_actor_4phase = self.actor_4phase(obs[four_phases_idxs], rnn_states_actor[four_phases_idxs], masks[four_phases_idxs], available_actions[four_phases_idxs], deterministic)
        values_3phase, rnn_states_critic_3phase = self.critic_3phase(share_obs[three_phases_idxs], rnn_states_critic[three_phases_idxs], masks[three_phases_idxs])
        values_4phase, rnn_states_critic_4phase = self.critic_4phase(share_obs[four_phases_idxs], rnn_states_critic[four_phases_idxs], masks[four_phases_idxs])
        return (values_3phase, values_4phase), (action_3phase, action_4phase), (action_log_probs_3phase, action_log_probs_4phase), (rnn_states_actor_3phase, rnn_states_actor_4phase), (rnn_states_critic_3phase, rnn_states_critic_4phase)

    def get_values(self, share_obs, rnn_states_critic, masks, three_phases_idxs=None, four_phases_idxs=None):
        values_3phase, _ = self.critic_3phase(share_obs[three_phases_idxs], rnn_states_critic[three_phases_idxs], masks[three_phases_idxs])
        values_4phase, _ = self.critic_4phase(share_obs[four_phases_idxs], rnn_states_critic[four_phases_idxs], masks[four_phases_idxs])
        return (values_3phase, values_4phase)

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None, three_phases_idxs=None, four_phases_idxs=None):
        action_log_probs_3phase, dist_entropy_3phase, policy_values_3phase = self.actor_3phase.evaluate_actions(obs[three_phases_idxs], rnn_states_actor[three_phases_idxs], action[three_phases_idxs], masks[three_phases_idxs], available_actions[three_phases_idxs], active_masks[three_phases_idxs])
        action_log_probs_4phase, dist_entropy_4phase, policy_values_4phase = self.actor_4phase.evaluate_actions(obs[four_phases_idxs], rnn_states_actor[four_phases_idxs], action[four_phases_idxs], masks[four_phases_idxs], available_actions[four_phases_idxs], active_masks[four_phases_idxs])
        values_3phase, _ = self.critic_3phase(share_obs[three_phases_idxs], rnn_states_critic[three_phases_idxs], masks[three_phases_idxs])
        values_4phase, _ = self.critic_4phase(share_obs[four_phases_idxs], rnn_states_critic[four_phases_idxs], masks[four_phases_idxs])
        return (values_3phase, values_4phase), (action_log_probs_3phase, action_log_probs_4phase), (dist_entropy_3phase, dist_entropy_4phase), (policy_values_3phase, policy_values_4phase)

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, three_phases_idxs=None, four_phases_idxs=None):
        actions_3phase, _, rnn_states_actor_3phase = self.actor_3phase(obs[three_phases_idxs], rnn_states_actor[three_phases_idxs], masks[three_phases_idxs], available_actions[three_phases_idxs], deterministic)
        actions_4phase, _, rnn_states_actor_4phase = self.actor_4phase(obs[four_phases_idxs], rnn_states_actor[four_phases_idxs], masks[four_phases_idxs], available_actions[four_phases_idxs], deterministic)
        return (actions_3phase, actions_4phase), (rnn_states_actor_3phase, rnn_states_actor_4phase)