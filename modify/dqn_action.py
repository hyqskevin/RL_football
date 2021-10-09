# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np
import torch

from util import GetRawObservations

ball_pos = [0, 0, 0]
last_dist = 0
player_pos = [0, 0, 0]


def action_modify(obs, action):
    global ball_pos
    global last_dist
    global player_pos
    get_obs = GetRawObservations(obs)
    team, pos, ball_direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    active_player = left_team[get_obs.get_player()]
    first_offense_player = max([player[0] for player in left_team])

    if (team == 0) and (active_player[0] > 0.6) and \
            (abs(active_player[1]) < 0.25) and \
            (ball_direction[0] > 0):
        modified_action = torch.IntTensor([[12]])
        return modified_action
    else:
        return action
