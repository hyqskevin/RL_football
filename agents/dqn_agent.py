# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
from util import ReplayBuffer
from nn.nn_layer import DQN


class DQNAgent:
    def __init__(self, num_actions=19, max_memory=10000, batch_size=64, gamma=0.9):
        super(DQNAgent, self).__init__()

        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state'))

        # define policy network to update Q function
        # define target network to compute TD target y_t
        self.policy_net = DQN(num_actions)
        self.target_net = DQN(num_actions)

        if torch.cuda.is_available():
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
        self.memory_buffer = ReplayBuffer(max_memory)
        self.steps_done = 0
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma

    def select_action(self, state):
        sample = random.random()
        eps_start = 0.9
        eps_end = 0.0
        eps_decay = 500
        eps_threshold = eps_end + (eps_start - eps_end) * \
                        math.exp(-1. * self.steps_done / eps_decay)
        # eps = args.epsilon - self.steps_done * 0.01
        if eps_threshold < 0.01:
            eps_threshold = 0.01
        self.steps_done += 1

        if sample > eps_threshold:
            # use policy network to choose a*
            with torch.no_grad():
                action_list = self.policy_net(state)
                action = action_list.max(1)[1].view(1, 1)
                # if 8 < action < 13:
                #     print('epsilon:{}, action num choose: {}'.format(eps_threshold, action))
                return action
        else:
            greedy_action = torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)
            # print('epsilon:{}, greedy_action'.format(eps_threshold, greedy_action))
            return greedy_action

    def optimize_model(self):
        # collect enough experience data
        if len(self.memory_buffer) < self.batch_size:
            return

        # get train batch (state, action, reward) from replay buffer
        trans_tuple = self.memory_buffer.sample(self.batch_size)
        batch = self.Transition(*zip(*trans_tuple))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(batch.next_state)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # compute Q(S_t, a) in policy net and gather action in terms of columns in action_batch
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q*(s_{t+1}, A) in target net for all next states.
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # next_state_values = self.target_net(next_state_batch)

        # Compute y = r + \gamma * max_a Q(s', a)
        expected_q_values = (next_state_values * self.gamma) + reward_batch
        # expected_q_values = torch.cat(
        #     tuple(reward + self.gamma * torch.max(q) for reward, q in
        #           zip(reward_batch, next_state_values)))

        # Compute loss and optimize model
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
