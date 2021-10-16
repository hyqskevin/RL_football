# -*- coding: utf-8 -*-
# @Author  : kevin_w

import gc
import os
import math
import numpy as np
import argparse
import torch
import gfootball.env as gf
from modify.ac_modify import reward_func
from itertools import count
from agents.a3c_agent import ActorCriticAgent
from util import trans_img, plot_training, save_data
from make_env import make_env
from multiprocessing_env import SubprocVecEnv


def train(path):

    envs = make_env(args.seed)
    envs = SubprocVecEnv(envs)
    num_actions = 19
    agent = ActorCriticAgent(num_actions)
    reward_list = []

    if torch.cuda.is_available():
        print('cuda:', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    for eps in range(args.episodes):
        # reset in each episode start
        obs = envs.reset()
        # obs = obs.transpose(0, 3, 1, 2)
        # image = trans_img(obs[0]['frame'])
        state = torch.FloatTensor(obs)

        if torch.cuda.is_available():
            state = state.cuda()

        eps_reward = 0
        for t in count():
            # get next_state, reward
            action = agent.select_action(state)
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())

            # next_obs = next_obs.transpose(0, 3, 1, 2)
            next_state = torch.FloatTensor(next_obs)

            eps_reward += np.sum(reward)
            reward = torch.FloatTensor(reward).unsqueeze(1)
            mask = torch.FloatTensor(1 - done).unsqueeze(1)

            if torch.cuda.is_available():
                # action = action.cuda()
                reward = reward.cuda()
                next_state = next_state.cuda()
                mask = mask.cuda()

            # agent.actions.append(agent.ActionTuple(
            #     math.log(action),
            #     state_value
            # ))
            agent.rewards.append(reward)
            agent.mask.append(mask)

            print('in step{}: action:{}, reward:{}'.format(t, action, torch.sum(reward)))

            if t > 1000:
                break

        reward_list.append(eps_reward)
        agent.optimize_model(t)

        # plot training rewards
        plot_training(reward_list, path)

        save_data(reward_list, 'a3c_reward', 'a3c_reward.xlsx')

        # save model
        if eps % 200 == 0 and eps > 0:
            model_path = './' + str(eps) + '_a3c_net.pkl'
            # os.makedirs(model_path, exist_ok=True)
            torch.save(agent.net.state_dict(), model_path)

        print('episode {}: last time {}, reward {}'.format(
            eps, t, reward_list
        ))

        # clean rewards and probs_log every episode
        agent.rewards = []
        agent.actions = []
        agent.values = []
        agent.mask = []

    print('reward list', reward_list)
    print('Complete')
    envs.close()


if __name__ == '__main__':
    # define parameter
    parser = argparse.ArgumentParser(description="Actor Critic example")
    parser.add_argument('--seed', type=int, default=1024, metavar='seed')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='gamma')
    parser.add_argument('--batch_size', type=int, default=128, metavar='batch')
    parser.add_argument('--episodes', type=int, default=5000, metavar='episodes')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')

    args = parser.parse_args()

    gc.collect()
    PATH = './A3C_plot/'
    os.makedirs(PATH, exist_ok=True)
    train(PATH)
