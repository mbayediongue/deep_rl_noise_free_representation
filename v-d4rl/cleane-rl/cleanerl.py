# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Predictor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Predictor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 119 * 119

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.linear(h)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class ContrastModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim), requires_grad=True)
        
        self.apply(utils.weight_init)

    def forward(self, z_pos):
        #h = self.trunk(obs)
        wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)

        return wz

class DrqCleanerAgent:
    def __init__(self, obs_shape, action_shape, max_action, num_protos, groups, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 offline=False, bc_weight=2.5, augmentation=RandomShiftsAug(pad=4),
                 use_bc=True, weight_contrastive_loss=1.):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.offline = offline
        self.bc_weight = bc_weight
        self.use_bc = use_bc
        self.weight_contrastive = weight_contrastive_loss

        # models
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(feature_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.predictor = Predictor(feature_dim, *action_shape, max_action).to(device)
        self.encoder_target = Encoder(obs_shape, feature_dim).to(device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        #self.Contrast = ContrastModule(feature_dim).to(device)
        self.critic = Critic(feature_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(feature_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        
        # optimizers
        self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters())+list(self.Contrast.parameters()), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic and encoder (Note that the encoder will be also optimized
        #  by 2 other losses. See method 'update_encoder' for more details)
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = -Q.mean()

        actor_loss = actor_policy_improvement_loss

        # offline BC Loss
        if self.offline:
            actor_bc_loss = F.mse_loss(action, behavioural_action)
            # Eq. 5 of arXiv:2106.06860
            lam = self.bc_weight / Q.detach().abs().mean()
            if self.use_bc:
                actor_loss = actor_policy_improvement_loss * lam + actor_bc_loss
            else:
                actor_loss = actor_policy_improvement_loss * lam

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.offline:
                metrics['actor_bc_loss'] = actor_bc_loss.item()

        return metrics
    
    

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out
    
    def update_encoder(self, obs1_k, obs2_k, action1, action2):
        metrics = dict()
        
        # encode the 2 observations
        obs1_k = self.encode(obs1_k)
        obs2_k = self.encode(obs2_k, ema=True)

        #-------------- Behavior loss---------------------------------------------
        behavioural_action = self.predictor(obs1_k)
        behavioural_action = F.normalize(behavioural_action, dim=1, p=2)
        action_ = F.normalize(action1, dim=1, p=2)
        
        behavioural_loss = F.mse_loss(action_, behavioural_action)

        #----------------contrastive loss-----------------------------------
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_traget(obs1_k, action1)
            V_obs1 = torch.min(target_Q1, target_Q2)

            target_Q1, target_Q2  = self.critic_target(obs2_k, action2)
            V_obs2 = torch.min(target_Q1, target_Q2)

        # Relative difference between the two state values function
        relative_diff_V = torch.abs(V_obs1-V_obs2)/(1e-8+torch.abs(V_obs1)+torch.asb(V_obs2))
        relative_diff_V = torch.exp(-5.*relative_diff_V)
        obs1_k = F.normalize(obs1_k, dim=1, p=2)
        obs1_k = F.normalize(obs2_k, dim=1, p=2)
        mse_ecodings = F.mse_loss(obs1_k, obs2_k)
        contrastive_loss = relative_diff_V*mse_ecodings - (1-relative_diff_V)*mse_ecodings
        
        loss = behavioural_loss + self.weight_constrastive*contrastive_loss 
        
        if self.use_tb:
            metrics['encoder_behavior_loss'] = behavioural_loss.item()
            metrics['encoder_contrastive_loss'] = contrastive_loss.item()

        self.encoder_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        return metrics
            
          
    # def update_encoder(self, obs, obs_k, k, step, action):
    #     metrics = dict()
        
    #     obs = self.encoder(obs)
    #     obs_k = self.encoder(obs_k)
    #     # k = self.k_embedding(k)
    #     obs_cat = torch.cat((obs, obs_k), dim = 1)
        
    #     behavioural_action = self.predictor(obs_cat)
        
    #     behavioural_action = F.normalize(behavioural_action, dim=1, p=2)
    #     action_ = F.normalize(action, dim=1, p=2)
        
    #     encoder_loss = F.mse_loss(action_, behavioural_action)

    #     # optimize actor and encoder
    #     self.encoder_opt.zero_grad(set_to_none=True)
    #     encoder_loss.backward()
    #     self.encoder_opt.step()
    #     return metrics
    
    def pretrain(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        #(obs_k_1, act, rew, dis, nobs, obs_k_2)
        obs_k2, action, reward, discount, next_obs, obs_k1, actions2 = utils.to_torch(
            batch, self.device)
        pos = torch.clone(obs)
        # augment
        obs = self.aug(obs.float())
        pos = self.aug(pos.float())
        metrics.update(self.update_encoder(obs, pos))

        return metrics
    
    
    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        #(obs1_k, act, rew, dis, nobs, obs2_k)
        obs, action, reward, discount, next_obs, obs2_k, action2 = utils.to_torch(
            batch, self.device)

        # augment
        # obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        
        pos = torch.clone(obs)
        # augment
        obs = self.aug(obs.float())
        pos = self.aug(pos.float())
        # metrics.update(self.update_encoder(obs, pos))
        
        
        # encode
        obs = self.encoder(obs)
        
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()


        # self.update_encoder(obs, step, action)
        self.update_encoder(obs, obs2_k, action, action2)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        if self.offline:
            metrics.update(self.update_actor(obs.detach(), step, action.detach()))
        else:
            metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.critic_target_tau)

        return metrics
