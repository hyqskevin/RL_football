# -*- coding: utf-8 -*-
# @Author  : kevin_w

import gym
import gfootball.env as gf
from util import TransEnv


def make_env(seed):
    # env1 = gf.create_environment(
    #     env_name='academy_empty_goal_close',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env2 = gf.create_environment(
    #     env_name='academy_empty_goal',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env3 = gf.create_environment(
    #     env_name='academy_run_to_score',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env4 = gf.create_environment(
    #     env_name='academy_run_to_score_with_keeper',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env5 = gf.create_environment(
    #     env_name='academy_pass_and_shoot_with_keeper',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env6 = gf.create_environment(
    #     env_name='academy_corner',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env7 = gf.create_environment(
    #     env_name='academy_3_vs_1_with_keeper',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )
    # env8 = gf.create_environment(
    #     env_name='academy_single_goal_versus_lazy',
    #     stacked=False,
    #     representation='extracted',
    #     rewards='scoring,checkpoint',
    #     write_goal_dumps=False,
    #     write_full_episode_dumps=False,
    #     render=False,
    #     write_video=False,
    #     # logdir=os.path.join(logger.get_dir(), "model.pkl"),
    #     logdir='./',
    #     extra_players=None,
    #     number_of_left_players_agent_controls=1,
    #     number_of_right_players_agent_controls=0
    # )

    env1 = TransEnv(gym.make('GFootball-academy_empty_goal_close-SMM-v0'))
    env2 = TransEnv(gym.make('GFootball-academy_empty_goal-SMM-v0'))
    env3 = TransEnv(gym.make('GFootball-academy_run_to_score-SMM-v0'))
    env4 = TransEnv(gym.make('GFootball-academy_run_to_score_with_keeper-SMM-v0'))
    env5 = TransEnv(gym.make('GFootball-academy_pass_and_shoot_with_keeper-SMM-v0'))
    env6 = TransEnv(gym.make('GFootball-academy_corner-SMM-v0'))
    env7 = TransEnv(gym.make('GFootball-academy_3_vs_1_with_keeper-SMM-v0'))
    env8 = TransEnv(gym.make('GFootball-academy_single_goal_versus_lazy-SMM-v0'))

    envs = [env1, env2, env3, env4, env5, env6, env7, env8]
    return envs
