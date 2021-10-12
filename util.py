# -*- coding: utf-8 -*-
# @Author  : kevin_w

import gym
import cv2
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt


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


# plot rewards
def plot_training(rewards, path):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.cla()
    ax.grid()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(rewards)
    run_time = len(rewards)

    # path = './AC_CartPole-v0/' + str(RunTime) + '.jpg'
    path = path + str(run_time) + '.jpg'
    if run_time % 100 == 0:
        plt.savefig(path)
    plt.pause(0.000001)
