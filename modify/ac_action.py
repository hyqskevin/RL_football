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
    team, pos, direction, player = get_obs.get_ball_info()
    left_team, right_team = get_obs.get_team_position()
    active_player = left_team[get_obs.get_player()]

    if (active_player[0] > 0.6) and (abs(active_player[1]) < 0.3):
        action = torch.IntTensor([12])
        print('modified shoot')
        return action
    else:
        return action
