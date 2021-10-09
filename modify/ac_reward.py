# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
from util import GetRawObservations

ball_pos = [0, 0, 0]
last_team = 0
last_dist = 0
player_pos = [0, 0, 0]
e = np.finfo(np.float32).eps.item()


# design a reward function based on raw info
def reward_func(obs, score, action):
    global ball_pos
    global last_team
    global last_dist
    global player_pos
    global e
    get_obs = GetRawObservations(obs)
    team, pos, direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    active_player = left_team[get_obs.get_player()]
    reward = 0

    # if opponents control, reward -
    if team == 1:
        reward -= 0.1
    if team == 1 and last_team != 1:
        reward -= 0.5
        last_team = 1
    if team == 0 and last_team != 0:
        # reward += 0.5
        last_team = 0

    # if ball outside the playground
    if team == 0 and \
            ((pos[0] <= -1) or (pos[0] >= 1) or (pos[1] >= 0.42) or (pos[1] <= -0.42)):
        reward -= 1
        print('outside punishment')

    # run to the ball and get control
    distance = math.sqrt((pos[0] - active_player[0])**2 + (pos[1] - active_player[1])**2)
    if (last_team != 0) and (distance-last_dist > 0.01):
        reward -= distance

    # offense
    if pos[1] > 0:
        if (team == 0) and (direction[0] > 0) and (direction[1] < 0) and (active_player[0] > -0.7):
            distance = math.sqrt((pos[0] - 1) ** 2 + (pos[1] - 0) ** 2)
            if last_dist - distance > 0.1:
                reward += (1.2 - distance)
    else:
        if (team == 0) and (direction[0] > 0) and (direction[1] > 0) and (active_player[0] > -0.7):
            distance = math.sqrt((pos[0] - 1) ** 2 + (pos[1] - 0) ** 2)
            if last_dist - distance > 0.1:
                reward += (1.2 - distance)

    # except control
    if (team == 0) and active_player[0] > 0.7 and (action == 12):
        reward += 1
        print('shoot opportunity')

    # score the goal +-5
    reward += score*5

    # update record
    ball_pos = pos
    last_dist = distance
    player_pos = active_player
    return reward
