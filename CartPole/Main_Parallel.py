import gymnasium as gym
import numpy as np
import random
import time
import math

from sklearn.preprocessing import KBinsDiscretizer

class CartPole():

    def __init__(self):
        self.envs = self.create_environment(human=False)

        self.episodes = 1000
        self.done = False

        self.bins = (6, 12)
        self.lower_bounds = [self.envs.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [self.envs.observation_space.high[2], math.radians(50)]

        self.Q_table = np.zeros(self.bins + (self.envs.action_space.n,))

    def create_environment(self, human=True):
        render = "human" if human else "rgb_array"
        return gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
    

game = CartPole()