# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
from util import GetRawObservations


ball_pos = [0, 0, 0]
player_pos = [0, 0, 0]
last_team = -1
last_dist = 0
e = np.finfo(np.float32).eps.item()


# design a reward function based on raw info
def reward_func(obs, score, action):
    global ball_pos
    global last_team
    global last_dist
    global player_pos
    global e
    get_obs = GetRawObservations(obs)
    team, pos, ball_direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    team_direction = get_obs.get_team_direction()
    active_player_direction = team_direction[get_obs.get_player()]
    active_player = left_team[get_obs.get_player()]
    reward = 0
    distance = 0
    first_offense_player = max([player[0] for player in left_team])
    ball_player_distance = math.sqrt((active_player[0] - pos[0])**2 + (active_player[1] - pos[1])**2)

    # if opponents control, reward -
    if team == 1:
        reward -= 0.1
    if team == 1 and last_team != 1:
        reward -= 0.1
        last_team = 1
    if team == 0 and last_team != 0:
        # reward += 0.5
        last_team = 0

    # if ball outside the playground
    if team == 0 and \
            ((pos[0] <= -1) or (pos[0] >= 1) or (pos[1] >= 0.42) or (pos[1] <= -0.42)):
        reward -= 0.5
        print('outside punishment')

    # run to the ball and get control
    # distance = math.sqrt((pos[0] - active_player[0])**2 + (pos[1] - active_player[1])**2)
    # if (last_team != 0) and (distance-last_dist > 0.0):
    #     reward -= 0.1

    # action limit
    if (team != 0) and (9 <= action <= 11):
        reward -= 0.1
        # print('catch punishment')
    if (team == 0) and (active_player_direction[0] > 0) and \
            (ball_direction[0] > 0) and \
            ((action == 13) or (action == 17)):
        reward += 0.5
        print('dribble reward')
    if (team == 0) and (active_player_direction[0] > 0) and \
            ball_player_distance < 0.05 and \
            ((pos[0] - ball_pos[0]) > 0):
        reward += 0.1
        print('controlled reward')
    if (team == 0) and active_player[0] >= first_offense_player:
        if (ball_direction[0] < 0) or \
                (active_player_direction[0] < 0) or \
                ((pos[0] - ball_pos[0]) < 0.0):
            reward -= 0.1
        # if (ball_pos[0] - pos[0]) < 0.1:
        #     reward -= 0.5
            print('pass behind punishment')
    if (active_player[0] < 0.6) and (action == 12):
        reward -= 0.1
        # print('shoot punishment')

    # offense
    if pos[1] > 0:
        if (team == 0) and ((pos[0] - ball_pos[0]) > 0) and \
                ((ball_pos[1] - pos[1]) > 0) and (active_player[0] > -0.7):
            distance = math.sqrt((active_player[0] - 1) ** 2 + (active_player[1] - 0) ** 2)
            if last_dist - distance > 0.0:
                reward += (2 - distance)
                print('move reward')
    elif pos[1] < 0:
        if (team == 0) and ((pos[0] - ball_pos[0]) > 0) and \
                ((ball_pos[1] - pos[1]) < 0) and (active_player[0] > -0.7):
            distance = math.sqrt((active_player[0] - 1) ** 2 + (active_player[1] - 0) ** 2)
            if last_dist - distance > 0.0:
                reward += (2 - distance)
                print('move reward')

    # except control
    # if (team == 0) and (direction[0] > 0) and (action == 13 or action == 17):
    #     reward += 0.1
    if (last_team == 0) and active_player[0] > 0.7 and abs(active_player[1]) < 0.2 and (action == 12):
        reward += 1
        print('shoot opportunity')

    # score the goal +-5
    reward += score*50

    # update record
    ball_pos = pos
    last_dist = distance
    player_pos = active_player
    return reward
