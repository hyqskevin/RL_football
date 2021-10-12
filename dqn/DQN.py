# -*- coding: utf-8 -*-
# @Author  : kevin_w

import gc
import os
import argparse
import torch
import gfootball.env as gf
from modify.dqn_modify import reward_func
from itertools import count
from agents.dqn_agent import DQNAgent
from util import trans_img, plot_training


def make_env():
    env = gf.create_environment(
        env_name='11_vs_11_easy_stochastic',
        stacked=False,
        representation='raw',
        rewards='scoring',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=True,
        write_video=False,
        # logdir=os.path.join(logger.get_dir(), "model.pkl"),
        logdir='./',
        extra_players=None,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0
    )
    # env = TransEnv(env)
    env.seed(args.seed)
    return env


def train(path):

    env = make_env()
    num_actions = env.action_space.n
    agent = DQNAgent(num_actions)
    reward_list = []

    if torch.cuda.is_available():
        print('cuda:', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # # test in one step
    # obs = env.reset()
    # # state = obs[0]['frame'].transpose(2, 0, 1)
    # # state = obs[0]['frame']
    # # state = trans_img(state)
    # image = trans_img(obs[0]['frame'])
    # image = torch.FloatTensor([image])
    # state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :].squeeze(0)
    # state = state.cuda()
    # print(state.shape)

    # action = agent.select_action(state)
    # next_obs, reward, done, _ = env.step(action.item())
    # next_state = trans_img(next_obs[0]['frame'])
    # reward = reward_func(next_obs, reward, action.item())
    # state = next_state
    # print(action.item(), reward)

    for eps in range(args.episodes):
        # reset in each episode start
        obs = env.reset()
        image = trans_img(obs[0]['frame'])
        state = torch.FloatTensor([image])
        next_obs = obs

        if torch.cuda.is_available():
            state = state.cuda()

        eps_reward = 0
        for t in count():
            # get next_state, reward
            action = agent.select_action(state, next_obs, eps)
            next_obs, reward, done, _ = env.step(action.item())
            next_img = trans_img(next_obs[0]['frame'])
            next_state = torch.FloatTensor([next_img])
            reward = reward_func(next_obs, reward, action.item())
            eps_reward += reward
            reward = torch.FloatTensor([reward])

            if torch.cuda.is_available():
                action = action.cuda()
                reward = reward.cuda()
                next_state = next_state.cuda()

            # Store the transition in memory and update agent
            agent.memory_buffer.push(state, action, reward, next_state)
            agent.optimize_model()

            if reward > 0:
                print('in step{}: action:{}, reward:{}'.format(t, action, reward))

            if not done:
                state = next_state
            else:
                state = None
            if done or t > 1500:
                break

        reward_list.append(eps_reward)
        # plot training rewards
        plot_training(reward_list, path)

        # save model
        if eps % 200 == 0 and eps > 0:
            # model_path = './dqn/' + str(eps) + '/'
            # os.makedirs(model_path, exist_ok=True)
            policy_path = './' + str(eps) + 'dqn_policy_net.pkl'
            target_path = './' + str(eps) + 'dqn_target_net.pkl'
            torch.save(agent.policy_net.state_dict(), policy_path)
            torch.save(agent.target_net.state_dict(), target_path)

        print('episode {}: last time {}, reward {}'.format(
            eps, t, eps_reward
        ))
        # update in each update interval
        if eps % args.update_interval == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print('reward list', reward_list)
    print('Complete')
    env.close()


if __name__ == '__main__':
    # define parameter
    parser = argparse.ArgumentParser(description="DQN example")
    parser.add_argument('--seed', type=int, default=1024, metavar='seed')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')
    parser.add_argument('--gamma', type=float, default=0.6, metavar='gamma')
    parser.add_argument('--epsilon', type=float, default=0.99, metavar='epsilon')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch')
    parser.add_argument('--max_memory', type=int, default=10000, metavar='max memory')
    parser.add_argument('--episodes', type=int, default=5000, metavar='episodes')
    parser.add_argument('--update_interval', type=int, default=100, metavar='target update',
                        help='interval between each target network update')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')

    args = parser.parse_args()

    PATH = './dqn/DQN_plot/'
    os.makedirs(PATH, exist_ok=True)
    gc.collect()
    train(PATH)
