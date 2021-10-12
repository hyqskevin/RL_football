# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from util import ReplayBuffer
from nn.nn_layer import PPOActor, PPOCritic
from modify.ac_modify import action_modify


class PPOAgent:
    def __init__(
            self,
            num_actions=19,
            max_memory=10000,
            batch_size=64,
            gamma=0.9,
            ppo=10,
            grad_norm=0.5,
            clip=0.2
    ):
        super(PPOAgent, self).__init__()

        self.ActionTransition = namedtuple(
            'Transition',
            ['state', 'action', 'value', 'log_prob', 'reward', 'next_state']
        )

        # self.ActionTuple = namedtuple('Action', ['log_prob', 'value'])

        self.actor_net = PPOActor(num_actions)
        self.critic_net = PPOCritic()
        if torch.cuda.is_available():
            self.actor_net.cuda()
            self.critic_net.cuda()

        self.actor_optimizer = optim.Adam(self.actor_net.parameters())
        self.critic_optimizer = optim.Adam(self.critic_net.parameters())
        # self.criterion = nn.SmoothL1Loss()
        # self.steps_done = 0
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.ppo_update_time = ppo
        self.max_grad_norm = grad_norm
        self.clip_param = clip
        self.memory_buffer = ReplayBuffer(max_memory)
        self.writer = SummaryWriter('./tensorboardX')
        self.training_step = 0

    def select_action(self, state, obs):

        with torch.no_grad():
            prob = self.actor_net(state)

        c = Categorical(prob)
        sample_action = c.sample()
        action_prob = prob[:, sample_action.item()].item()

        # manual modify the action
        # sample_action = action_modify(obs, sample_action)

        if torch.cuda.is_available():
            sample_action = sample_action.cuda()
            action_prob = action_prob.cuda()

        # self.actions.append(self.ActionTuple(
        #     c.log_prob(sample_action),
        #     state_value))

        return sample_action, prob[:, sample_action.item()].item()

    def get_value(self, state):
        with torch.no_grad():
            state_value = self.critic_net(state)

        if torch.cuda.is_available():
            state_value = state_value.cuda()

        return state_value

    def loss_function(self, eps):

        # get train batch (state, action, reward) from replay buffer
        trans_tuple = self.memory_buffer.sample(self.batch_size)
        batch = self.ActionTransition(*zip(*trans_tuple))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        value_batch = torch.cat(batch.value)
        reward_batch = torch.cat(batch.reward)
        log_prob_batch = torch.cat(batch.log_prob)

        R = 0
        total_rewards = []
        eps = np.finfo(np.float32).eps.item()

        for r in reward_batch[::-1]:
            R = r + self.gamma * R
            total_rewards.insert(0, R)

        total_rewards = torch.tensor(total_rewards)
        total_rewards = (total_rewards - total_rewards.mean()) / (total_rewards.std() + eps)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(batch))), self.batch_size, False):

                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(eps, self.training_step))

                # advantage function
                U = total_rewards[index].view(-1, 1)
                V = value_batch[index]
                # V = self.critic_net(state[index])
                advantage = U - V

                # epoch iteration, PPO core
                action_prob = self.actor_net(state_batch[index]).gather(1, action_batch[index])  # new policy
                ratio = (action_prob / log_prob_batch[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # return policy_loss, value_loss
                policy_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN descent
                self.writer.add_scalar('loss/action_loss', policy_loss, global_step=self.training_step)

                value_loss = func.mse_loss(U, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)

                self.training_step += 1

                return policy_loss, value_loss

    def optimize_model(self, eps):

        # collect enough experience data
        if len(self.memory_buffer) < self.batch_size:
            return

        policy_loss, value_loss = self.loss_function(eps)

        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # clear experience
        del self.memory_buffer[:]
