# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
from util import GetRawObservations

ball_pos = [0, 0, 0]
last_team = -1


# design a reward function based on raw info
def reward_func(obs, score, action):
    global ball_pos
    global last_team
    get_obs = GetRawObservations(obs)
    team, pos, direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    active_player = left_team[get_obs.get_player()]
    reward = 0

    # if opponents control, reward -
    if team == 1:
        reward -= 0.1
    if team == 1 and last_team == 0:
        reward -= 0.5
        last_team = 1
    if team == 0 and last_team == 1:
        reward += 0.5
        last_team = 0

    # run to the ball and get control
    distance = math.sqrt((pos[0] - active_player[0])**2 + (pos[1] - active_player[1])**2)
    if team != 0 and distance > 0.05:
        reward -= distance

    # offense
    if active_player[0] > -0.7 and (team == 0) and ((pos[0] - ball_pos[0]) > 0):
        reward += (0.6 * pos[0] + 0.4 * (abs(0.42 - pos[1])))

    # except control
    if (team == 0) and (action == 17):
        reward += 0.1
    if (team != 0) and (9 <= action <= 11):
        reward -= 10
    if (direction[0]) < 0 and (9 <= action <= 10):
        reward -= 10
    if pos[0] > 0.8 and (9 <= action <= 10):
        reward -= 10
    if pos[0] > 0.9 and abs(pos[1]) < 0.1 and (action == 12):
        reward += 1

    # score the goal +-5
    reward += score*5

    ball_pos = pos
    return reward
