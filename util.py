# -*- coding: utf-8 -*-
# @Author  : kevin_w

import gym
import cv2
import numpy as np
import random
from collections import namedtuple, deque


# resize and transpose observation
class TransEnv(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.height = 72
        self.width = 128
        self.channel = 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.channel, self.height, self.width),
            dtype=np.uint8
        )

    def observation(self, observation):
        obs = cv2.resize(observation,
                         (self.width, self.height),
                         interpolation=cv2.INTER_AREA
                         )
        return obs.reshape(self.observation_space.low.shape)


# get raw observatios


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


class GetRawObservations:
    def __init__(self, obs):
        self.ball_owned_team = obs[0]['ball_owned_team']
        self.ball_position = obs[0]['ball']
        self.ball_owned_player = obs[0]['ball_owned_player']
        self.ball_direction = obs[0]['ball_direction']
        self.left_team_position = obs[0]['left_team']
        self.right_team_position = obs[0]['right_team']
        self.designated_player = obs[0]['designated']

    def get_ball_info(self):
        return self.ball_owned_team, self.ball_position, self.ball_direction, self.ball_owned_player

    def get_team_position(self):
        return self.left_team_position, self.right_team_position

    def get_player(self):
        return self.designated_player


# build replay buffer in DQN
class ReplayBuffer(object):
    def __init__(self, capacity):
        # define the max capacity of memory
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'reward', 'next_state'))

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, bach_size):
        return random.sample(self.memory, bach_size)


# resize and transpose image
def trans_img(image):
    height = 72
    width = 128
    image = cv2.resize(image,
                       (width, height),
                       interpolation=cv2.INTER_AREA
                       )
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32) / 255
    return image
