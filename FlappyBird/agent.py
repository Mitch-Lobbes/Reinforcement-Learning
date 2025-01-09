import torch
import flappy_bird_gymnasium
import gymnasium
import itertools
import yaml
import random
import numpy as np
import os
import matplotlib
import datetime
import matplotlib.pyplot as plt
import argparse

from torch import nn
from DQN import DQN
from experience_replay import ReplayMemory

DATE_FORMAT = "%m-%d %H:%M:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots without a window
matplotlib.use('Agg')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Agent:

    def __init__(self, hyperparameter_set):
        self.config = yaml.safe_load(open("hyperparameters.yml"))
        self.config = self.config[hyperparameter_set]
        self.config['loss_fn'] = nn.MSELoss()
        self.config['optimizer'] = None


        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparameter_set}.png")


    def run(self, is_training=True, render=False):
        if is_training:
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training started\n"
            print(log_message)
            with open(self.LOG_FILE, "w") as log_file:
                log_file.write(log_message+"\n")

        #env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make(self.config["env_id"], render_mode="human" if render else None, **self.config.get("env_config", {}))

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_eisode = []

        policy_dqn = DQN(num_states, num_actions, self.config["fc1_nodes"]).to(device)

        if is_training:
            memory = ReplayMemory(self.config["replay_memory_size"])
            epsilon = self.config["epsilon_init"]

            target_dqn = DQN(num_states, num_actions, self.config["fc1_nodes"]).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.config['optimizer'] = torch.optim.Adam(policy_dqn.parameters(), lr=self.config["learning_rate"])

            step_count = 0
            epsilon_history = []
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float).to(device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.config["stop_on_reward"]):

                if is_training and random.random() < epsilon:
                    # Next action is random
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)
                else:
                    with torch.no_grad():
                        # Tensor of shape ([1,2,3,...]) -> ([[1,2,3,...]])
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float).to(device)
                reward = torch.tensor(reward, dtype=torch.float).to(device)

                if is_training:
                    # Store the transition in the replay memory
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1

                state = new_state

            rewards_per_eisode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: Episode {episode} - New best reward: {episode_reward}\n"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_message+"\n")
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                
                current_time = datetime.datetime.now()
                if current_time - last_graph_update_time > datetime.timedelta(seconds=10):
                    #self.save_graph(rewards_per_eisode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) >= self.config["mini_batch_size"]:
                    # Update the policy network
                    mini_batch = memory.sample(self.config["mini_batch_size"])
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.config["epsilon_decay"], self.config["epsilon_min"])
                    epsilon_history.append(epsilon)    

                    if step_count > self.config['network_sync_rate']:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))

        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):x+1])
        
        plt.subplots(121)
        plt.ylabel("Mean reward")
        plt.plot(mean_rewards)

        plt.subplots(122)
        plt.ylabel("Epsilon decay")
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        plt.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and seperate each element
        states, actions, new_states, rewards, terminations= zip(*mini_batch)

        # Stack tensors to create batch tensors
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)
        
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float).to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.config['gamma'] * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states) -> [[1,2,3], [4,5,6]]
                    .max(dim=1) -> torch.return_types.max(values=tensor([3,6]), indices=tensor([3,0,0,1]))
                    [0] -> tensor([3,6])
            '''

        # Calculate the current Q-values

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states) -> [[1,2,3], [4,5,6]]
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1)) ==>
                    .squeeze() ==>
        '''

        # Calculate the loss
        loss = self.config['loss_fn'](current_q, target_q)

        self.config['optimizer'].zero_grad()    # Reset the gradients
        loss.backward()                        # Calculate the gradients
        self.config['optimizer'].step()     # Update the weights


if __name__ == "__main__":
    # agent = Agent("cartpole1")
    # agent.run(is_training=True, render=True)

    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)