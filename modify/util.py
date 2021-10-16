# -*- coding: utf-8 -*-
# @Author  : kevin_w

import math
import numpy as np

action_count = 0
last_action = 0

"""
{'ball_owned_team': -1, 
'left_team_direction': array([[-8.56129418e-07,  5.13117108e-03],
       [ 0.00000000e+00, -0.00000000e+00]]), 
       'right_team_direction': array([[-9.44390649e-06, -5.47113712e-04]]), 
       'right_team': array([[-1.01103413, -0.00386697]]), 
       'left_team_yellow_card': array([False, False]), 
       'right_team_roles': array([0]), 
       'right_team_active': array([ True]), 
       'left_team': array([[-1.01103008,  0.0068209 ],
       [ 0.75827205, -0.        ]]), 
       'ball_rotation': array([ 0., -0.,  0.]), 
       'game_mode': 0, 'right_team_tired_factor': array([3.59416008e-05]), 
       'ball': array([ 0.76999998, -0.        ,  0.11059734]), 
       'score': [0, 0], 
       'left_team_roles': array([0, 1]), 
       'ball_owned_player': -1, 
       'left_team_tired_factor': array([0.00022763, 0.        ]), 
       'ball_direction': array([-0.        ,  0.        , -0.00191829]), 
       'left_team_active': array([ True,  True]), 
       'right_team_yellow_card': array([False]), 
       'steps_left': 400, 
       'designated': 1, 
       'active': 1, 
       'sticky_actions': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8), 
       'frame': array([[[ 93, 110,  78],
        [ 94, 111,  79],
"""


# get raw observations
class GetRawObservations:
    def __init__(self, obs):
        self.ball_owned_team = obs[0]['ball_owned_team']
        self.ball_position = obs[0]['ball']
        self.ball_owned_player = obs[0]['ball_owned_player']
        self.ball_direction = obs[0]['ball_direction']
        self.left_team_direction = obs[0]['left_team_direction']
        self.left_team_position = obs[0]['left_team']
        self.right_team_position = obs[0]['right_team']
        self.designated_player = obs[0]['designated']

    def get_ball_info(self):
        return self.ball_owned_team, self.ball_position, self.ball_direction, self.ball_owned_player

    def get_team_position(self):
        return self.left_team_position, self.right_team_position

    def get_team_direction(self):
        return self.left_team_direction

    def get_player(self):
        return self.designated_player


# count repeated action
def collect_action(action):
    global action_count
    global last_action

    if action == last_action:
        action_count += 1

    # if repeat over 10 step
    # modify will trigger to prevent fall into local minimum
    if action_count >= 20:
        action_count = 0
        last_action = action
        return action, True
    else:
        last_action = action
        return action, False
