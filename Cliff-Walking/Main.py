import gymnasium as gym
import numpy as np


class CliffWalking:

    def __init__(self):
        self.env = self.create_environment(human=True)
        self.episodes = 100
        self.done = False

        # Create Q-table matrix with all zeros of size (state_space x action_space)
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def create_environment(self, human=True):
        render = "human" if human else "rgb_array"
        return gym.make('CliffWalking-v0',render_mode=render)
    
    def simple_learning_rate(self, t, alpha_0=1.0, k=0.1, min_rate=0.01):
        return max(min_rate, alpha_0 / (1 + k * t))
    
    def simple_discount_rate(self):
        return 0.5
    
    def bellman_equation(self, t, reward, previous_state, new_state, action_taken):
        new_Q_value = self.Q_table[previous_state][action_taken] + self.simple_learning_rate(t) * (reward + self.simple_discount_rate() * np.max(self.Q_table[new_state]) - self.Q_table[previous_state][action_taken])
        return new_Q_value
    
    def save_best_model(self):
        np.save('best_Q_table.npy', self.Q_table)

    def run_best_model(self):

        best_Q_table = np.load('best_Q_table.npy')

        self.env = self.create_environment(human=True)
        state, _ = self.env.reset()
        self.done = False

        while not self.done:
            action = np.argmax(best_Q_table[state])
            state, _, self.done, _, _ = self.env.step(action)
            self.env.render()
    
    def start_training(self):
        
        for episode in range(self.episodes):
            print(f"\nEpisode: {episode}")
            self.done = False
            
            # Start / Reset the environment
            current_state, _ = self.env.reset()

            while not self.done:

                # Choose an action: pure-greedy
                action = np.argmax(self.Q_table[current_state])

                # Perform the action
                new_state, reward, done, trunc, prob = self.env.step(action)
            
                # Update Q-table
                new_q_value = self.bellman_equation(episode, reward, new_state, current_state, action)
                self.Q_table[current_state][action] = new_q_value

                # Update state
                current_state = new_state

                # Check if the episode is done
                self.done = done

                # Render the environment
                self.env.render()

        self.save_best_model()

game = CliffWalking()
game.start_training()
game.run_best_model()