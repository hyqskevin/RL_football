# -*- coding: utf-8 -*-
# @Author  : kevin_w
# test the google football environment

import gfootball.env as gf
from baselines import logger
from itertools import count
from util import TransEnv

""" 
Scenario name
1)  11_vs_11_easy_stochastic
2)  11_vs_11_hard_stochastic
3)  11_vs_11_stochastic
4)  academy_3_vs_1_with_keeper
5)  academy_corner
6)  academy_counterattack_easy
7)  academy_counterattack_hard
8)  academy_empty_goal_close
9)  academy_empty_goal
10) academy_pass_and_shoot_with_keeper
11) academy_run_pass_and_shoot_with_keeper
12) academy_run_to_score
13) academy_run_to_score_with_keeper
14) academy_single_goal_versus_lazy

representation
simple115, pixels, pixels_gray, extracted

modify
scoring, checkpoints
"""


# build football environment
def make_env():
    env = gf.create_environment(
        env_name='academy_run_to_score_with_keeper',
        stacked=False,
        representation='raw',
        rewards='scoring,checkpoints',
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
    return env


def test_env():
    env = make_env()
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    env.reset()
    for t in count():
        action = env.action_space.sample()  # random sample one action in action space
        new_obs, reward, done, info = env.step(action)
        print('action:{}, reward:{}, info:{}'.format(action, reward, info))
        if done:
            env.reset()
        if t >= 100:
            break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_env()
