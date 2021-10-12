# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from nn.nn_layer import ActorCritic
from modify.ac_modify import action_modify


class ActorCriticAgent:
    def __init__(self, num_actions, batch_size=64, gamma=0.9):
        super(ActorCriticAgent, self).__init__()

        self.ActionTuple = namedtuple('Action', ['log_prob', 'value'])

        self.net = ActorCritic(num_actions)
        if torch.cuda.is_available():
            self.net.cuda()

        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.steps_done = 0
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.rewards = []
        self.actions = []
        # TODO
        # add tensorboardX

    def select_action(self, state, obs, epo):

        with torch.no_grad():
            prob, state_value = self.net(state)

        c = Categorical(prob)
        sample_action = c.sample()

        # manual modify the action in 100 episodes
        # then agent fully control the decision
        if epo < 100:
            sample_action = action_modify(obs, sample_action)

        if torch.cuda.is_available():
            sample_action = sample_action.cuda()
            state_value = state_value.cuda()

        self.actions.append(self.ActionTuple(
            c.log_prob(sample_action),
            state_value))

        return sample_action

    def loss_function(self):
        R = 0
        saved_actions = self.actions
        policy_loss = []
        value_loss = []
        total_rewards = []
        eps = np.finfo(np.float32).eps.item()

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            total_rewards.insert(0, R)

        total_rewards = torch.tensor(total_rewards)
        total_rewards = (total_rewards - total_rewards.mean()) / (total_rewards.std() + eps)

        for (log_prob, state_value), U in zip(saved_actions, total_rewards):

            # value loss (critic)
            # y - U
            value_loss.append(self.criterion(state_value, torch.tensor([U]).cuda()))

            # policy loss (actor)
            # U - V(s), loss = - (U - V(S)) * log(\pi(A|S))
            policy_loss.append((-log_prob * (U - state_value.item())))

        return policy_loss, value_loss

    def optimize_model(self, count):
        policy_loss, value_loss = self.loss_function()
        self.optimizer.zero_grad()
        loss = (torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()) / count
        loss.backward()
        self.optimizer.step()
