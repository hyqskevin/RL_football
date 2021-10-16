# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
import torch
from modify.util import collect_action, GetRawObservations


ball_pos = [0, 0, 0]
player_pos = [0, 0, 0]
last_team = 0
last_dist_to_ball = 0
last_dist_to_goal = 0
e = np.finfo(np.float32).eps.item()


def action_modify(obs, action):
    get_obs = GetRawObservations(obs)
    team, pos, ball_direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    active_player = left_team[get_obs.get_player()]

    # modify it if the agent repeat one single action to prevent local minimum
    # dribble the ball for a long time
    # pass the ball to each other for a long time
    # take a shoot in a long distance to the goal

    a, is_in_local_min = collect_action(action.item())
    if (team == 0) and (0 <= a <= 8) and \
            is_in_local_min:
        modified_action = torch.IntTensor([[9]])
        return modified_action

    if (team == 0) and (9 <= a <= 11) and \
            is_in_local_min:
        modified_action = torch.IntTensor([[5]])
        print('modified action2')
        return modified_action

    if (active_player[0] <= 0.4) and a == 12:
        modified_action = torch.IntTensor([[5]])
        print('modified shoot1')
        return modified_action

    if (team == 0) and (active_player[0] > 0.6) and \
            (abs(active_player[1]) < 0.25) and \
            (ball_direction[0] > 0):
        modified_action = torch.IntTensor([[12]])
        print('modified shoot2')
        return modified_action
    else:
        return action


# design a reward function based on raw info
def reward_func(obs, score, action):
    global ball_pos
    global last_team
    global last_dist_to_ball
    global last_dist_to_goal
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
            ((pos[0] <= -1.02) or (pos[0] >= 1.02) or (pos[1] >= 0.42) or (pos[1] <= -0.42)):
        reward -= 0.5
        print('outside punishment')

    # run to the ball and get control
    distance_to_ball = math.sqrt((pos[0] - active_player[0])**2 + (pos[1] - active_player[1])**2)
    if (last_team != 0) and (distance-last_dist_to_ball > 0.01):
        reward -= 0.1

    # action limit
    if (team != 0) and (9 <= action <= 11):
        reward -= 0.1
        print('uncontrolled punishment')
    if (team == 0) and (active_player[0] < 0.7) and \
            active_player[0] >= first_offense_player:
        if (ball_direction[0] < 0) or \
                (active_player_direction[0] < 0) or \
                ((pos[0] - ball_pos[0]) < 0.0):
            reward -= 0.1
            print('pass behind punishment')
    if (team == 0) and (active_player_direction[0] > 0.01) and \
            (ball_direction[0] > 0.01) and \
            ((action == 13) or (action == 17)):
        reward += 0.5
        print('dribble reward')
    if (active_player[0] < 0.6) and (action == 12):
        reward -= 0.1
        print('shoot punishment')
    if (team == 0) and active_player[0] > 0.6 and \
            (abs(active_player[1]) < 0.3) and \
            (action == 12):
        reward += 1
        print('shoot opportunity')

    # offense
    distance = math.sqrt((active_player[0] - 1) ** 2 + (active_player[1] - 0) ** 2)
    if pos[1] > 0:
        if (team == 0) and ((pos[0] - ball_pos[0]) > 0) and \
                ((ball_pos[1] - pos[1]) > 0) and (active_player[0] > -0.7):
            if last_dist_to_goal - distance > 0.01:
                reward += (2 - distance)
                print('move reward')
            if (active_player_direction[0] > 0.001) and \
                    ball_player_distance < 0.05:
                reward += (2 - distance)
                print('controlled reward')
    elif pos[1] < 0:
        if (team == 0) and ((pos[0] - ball_pos[0]) > 0) and \
                ((ball_pos[1] - pos[1]) < 0) and (active_player[0] > -0.7):
            if last_dist_to_goal - distance > 0.01:
                reward += (2 - distance)
                print('move reward')
            if (active_player_direction[0] > 0.001) and \
                    ball_player_distance < 0.05:
                reward += (2 - distance)
                print('controlled reward')

    # score the goal +-5
    reward += score*50

    # update record
    ball_pos = pos
    last_dist_to_ball = distance_to_ball
    last_dist_to_goal = distance
    player_pos = active_player
    return reward
