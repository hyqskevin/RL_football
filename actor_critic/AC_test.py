# -*- coding: utf-8 -*-
# @Author  : kevin_w

import os
import argparse
import gfootball.env as gf
import torch
from torch.distributions import Categorical
from nn.nn_layer import ActorCritic
from itertools import count
from util import trans_img, plot_training, save_data
from modify.ac_modify import action_modify


class ACTest:
    def __init__(self, num_actions=19, batch_size=64, gamma=0.9):
        super(ACTest, self).__init__()

        self.net = ActorCritic(num_actions)

        if torch.cuda.is_available():
            self.net.cuda()
            self.net.load_state_dict(torch.load("{}/1000_ac_net.pkl".format(args.saved_path)))
        else:
            self.net.load_state_dict(torch.load("{}/1000_ac_net.pkl".format(args.saved_path)))

        self.net.eval()

    def select_action(self, state, obs):

        prob, state_value = self.net(state)

        c = Categorical(prob)
        sample_action = c.sample()
        sample_action = action_modify(obs, sample_action)

        if torch.cuda.is_available():
            sample_action = sample_action.cuda()

        return sample_action

# test environment
# 1)  11_vs_11_easy_stochastic
# 2)  11_vs_11_hard_stochastic
# 3)  11_vs_11_stochastic


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
    env.seed(args.seed)
    return env


def test(path):
    env = make_env()

    if torch.cuda.is_available():
        print('cuda:', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    model = ACTest()

    scores = []
    for match in range(10):
        obs = env.reset()
        image = trans_img(obs[0]['frame'])
        state = torch.FloatTensor([image])
        next_obs = obs

        if torch.cuda.is_available():
            state = state.cuda()

        score = 0
        for t in count():
            # get next_state, reward
            action = model.select_action(state, next_obs)
            next_obs, reward, done, _ = env.step(action.item())
            next_img = trans_img(next_obs[0]['frame'])
            next_state = torch.FloatTensor([next_img])
            score += reward

            if torch.cuda.is_available():
                next_state = next_state.cuda()

            if not done:
                state = next_state
            else:
                state = None
            if done:
                break
        scores.append(score)

        # plot training rewards
        plot_training(scores, path)

    # save reward list
    save_data(scores, 'test_score', 'test_score.xlsx')

    print('Complete')
    env.close()


if __name__ == '__main__':
    # define parameter
    parser = argparse.ArgumentParser(description="Actor Critic test")
    parser.add_argument('--seed', type=int, default=1024, metavar='seed')
    parser.add_argument('--saved_path', type=str, default='pkl_old', metavar='path')

    args = parser.parse_args()

    PATH = 'AC_plot_test/'
    os.makedirs(PATH, exist_ok=True)
    test(PATH)
