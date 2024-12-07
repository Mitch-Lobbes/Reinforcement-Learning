import gymnasium as gym
import numpy as np
import random
import time
import math

from sklearn.preprocessing import KBinsDiscretizer

env = gym.make('CartPole-v1',render_mode="human")
#env = gym.make('CartPole-v1',render_mode="rgb_array")



class CartPole():

    def __init__(self):
        self.env = self.create_environment(human=False)
        self.episodes = 1000
        self.done = False

        self.bins = (6, 12)
        self.lower_bounds = [self.env.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [self.env.observation_space.high[2], math.radians(50)]

        self.Q_table = np.zeros(self.bins + (self.env.action_space.n,))


    def create_environment(self, human=True):
        render = "human" if human else "rgb_array"
        return gym.make('CartPole-v1',render_mode=render)
        #return gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
    
    def discretizer(self, angle, pole_velocity):
        est = KBinsDiscretizer(n_bins=self.bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))
    
    def policy(self, state):
        return np.argmax(self.Q_table[state])
    
    def new_Q_value(self, reward, new_state, discount_factor=1):
        future_optimal_value = np.max(self.Q_table[new_state])
        learned_value = reward + discount_factor * future_optimal_value
        return learned_value
        
    def learning_rate(self, n, min_rate=0.01):
        return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))
    
    def exploration_rate(self, n, min_rate=0.1):
        return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))

    def start_training(self):
        
        for e in range(self.episodes):
            print(f"\nEpisode: {e}")
            self.done = False
            current_state, _ = self.env.reset()   
            angle, pole_velocity = current_state[2], current_state[3]
            #print(f"Start observation: {current_state}")

            current_state = self.discretizer(angle, pole_velocity)

            while not self.done:

                action = self.policy(current_state)
                #print(f'Action: {action}')

                if np.random.random() < self.exploration_rate(e):
                    action = self.env.action_space.sample()

                new_state, reward, self.done, _, _ = self.env.step(action)
                angle, pole_velocity = new_state[2], new_state[3]
                new_state = self.discretizer(angle, pole_velocity)

                lr = self.learning_rate(e)
                learnt_value = self.new_Q_value(reward, new_state)
                old_value = self.Q_table[current_state][action]
                self.Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value

                current_state = new_state

                self.env.render()

        self.save_best_model()

    def save_best_model(self):
        np.save('best_Q_table.npy', self.Q_table)

    def run_best_model(self):

        best_Q_table = np.load('best_Q_table.npy')

        self.env = self.create_environment(human=True)
        state, _ = self.env.reset()
        self.done = False

        while not self.done:
            angle, pole_velocity = state[2], state[3]
            state = self.discretizer(angle, pole_velocity)
            action = np.argmax(best_Q_table[state])
            state, _, self.done, _, _ = self.env.step(action)
            self.env.render()

        self.env.close()


game = CartPole()
game.start_training()
game.run_best_model()