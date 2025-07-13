import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import math
from utils.utils import *
from utils.nodes import *
from WorldEnvironment.SnakeGame.SnakeEnv import envSnake
from matplotlib import pyplot as plt
import random
from collections import deque


class PolicyNetworkModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.env = envSnake()
        self.input_dim = math.prod(self.env.observation_space)#input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = math.prod(self.env.action_space)#output_dim
        self.dropout = dropout
        self.MAX_EPOCHS = 5000
        self.DISCOUNT_FACTOR_MAX = 0.99
        self.DISCOUNT_FACTOR_MIN = 0.99
        self.DISCOUNT_FACTOR = self.DISCOUNT_FACTOR_MIN
        self.DISCOUNT_FACTOR_STEP = (self.DISCOUNT_FACTOR_MAX - self.DISCOUNT_FACTOR_MIN)/self.MAX_EPOCHS
        self.N_TRIALS = 25
        self.REWARD_THRESHOLD = 1000
        self.PRINT_INTERVAL = 10
        self.INPUT_DIM = math.prod(self.env.observation_space)#env.observation_space.shape[0]
        self.HIDDEN_DIM = 128
        self.OUTPUT_DIM = math.prod(self.env.action_space)#env.action_space.n
        self.DROPOUT = 0.5
        self.MAX_LOOP = 100
        self.START_LOOP = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.episode_returns = []

        self.Policy = PolicyNetworkCNN(self.input_dim,
                                    self.hidden_dim,
                                    self.output_dim,
                                    self.dropout).to(device=self.device,dtype=torch.float)
        
        self.LEARNING_RATE = 0.001
        self.optimizer = optim.Adam(self.Policy.parameters(), lr = self.LEARNING_RATE)
        
    def forward_pass(self):
        states = []
        log_prob_actions = []
        rewards = []
        entropys = []
        done = False
        episode_return = 0

        self.Policy.eval()
        observation, info = self.env.reset()
        n_loop = 0

        while not done and n_loop < self.MAX_LOOP:
            observation = torch.tensor(observation).to(device=self.device).unsqueeze(0).permute(0, 3, 1, 2)
            # print(observation[:,:,2])
            # break
            states.append(observation)
            # observation = torch.tensor(observation).to(device=self.device).flatten().unsqueeze(0)
            action_pred = self.Policy(observation)
            action_prob = F.softmax(action_pred, dim = -1)
            # print(action_prob)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()
            log_prob_action = dist.log_prob(action)
            entropy = dist.entropy()
            entropys.append(entropy[0])
            # print(entropy.size())
            # print(log_prob_action)
            # print(action)

            observation, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated
            # if reward == 1:
                # print("get eatting!!!!")

            log_prob_actions.append(log_prob_action)
            rewards.append(reward)
            episode_return += reward

            n_loop += 1

        log_prob_actions = torch.cat(log_prob_actions)
        stepwise_returns = calculate_stepwise_returns(rewards, self.DISCOUNT_FACTOR).to(device=self.device)
        # self.DISCOUNT_FACTOR += self.DISCOUNT_FACTOR_STEP

        return episode_return, stepwise_returns, log_prob_actions, states, entropys
    
    def update_policy(self, stepwise_returns, log_prob_actions, states, optimizer, entropys):
        stepwise_returns = stepwise_returns.detach()
        loss = calculate_loss(stepwise_returns, log_prob_actions)
        # loss = loss - 0.01 * (sum(entropys)/len(entropys))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # self.Policy.train()
        # for stepwise_return, log_prob_action, observation in zip(stepwise_returns, log_prob_actions, states):

        #     action_pred = self.Policy(observation)
        #     action_prob = F.softmax(action_pred, dim = -1)
        #     # print(action_prob)
        #     dist = distributions.Categorical(action_prob)
        #     action = dist.sample()
        #     log_prob_action = dist.log_prob(action)

        #     stepwise_return = stepwise_return.detach()
        #     loss = -(stepwise_return * log_prob_action)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()


        return loss.item()
    
    def traningM(self):
        for episode in range(1, self.MAX_EPOCHS+1):
            # self.MAX_LOOP = (episode // 10) + self.START_LOOP
            episode_return, stepwise_returns, log_prob_actions, states, entropys = self.forward_pass()
            # break
            loss = self.update_policy(stepwise_returns, log_prob_actions, states, self.optimizer, entropys)

            print(f"Episode: {episode} ==> Loss: {loss}")

            self.episode_returns.append(episode_return)
            mean_episode_return = np.mean(self.episode_returns[-self.N_TRIALS:])

            if episode % self.PRINT_INTERVAL == 0:
                print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')

            if mean_episode_return >= self.REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
    

class DQNAgent:
    def __init__(self):
        self.env = envSnake()
        obs_shape = self.env.observation_space
        n_actions = math.prod(self.env.action_space)

        self.INPUT_DIM = obs_shape # e.g. (3, 10, 10) -> C, H, W
        self.HIDDEN_DIM = 32
        self.OUTPUT_DIM = n_actions
        self.DROPOUT = 0.1
        self.LEARNING_RATE = 1e-4
        self.BATCH_SIZE = 64
        self.MEMORY_SIZE = 10000
        self.GAMMA = 0.99
        self.EPSILON_START = 1.0
        self.EPSILON_END = 0.1
        self.EPSILON_DECAY = 0.999945
        self.TARGET_UPDATE = 1000
        self.MAX_EPOCHS = 50000
        self.MAX_LOOP = 1000000000
        self.REWARD_THRESHOLD = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = self.EPSILON_START

        self.policy_net = DQNCNN(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.DROPOUT).to(self.device)
        self.target_net = DQNCNN(self.INPUT_DIM, self.HIDDEN_DIM, self.OUTPUT_DIM, self.DROPOUT).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LEARNING_RATE)

        self.episode_returns = []
        self.last_action = None
        self.opposite_action = {0: 1, 1: 0, 2: 3, 3: 2}  # LEFT <-> RIGHT, UP <-> DOWN


    # def select_action(self, state):
    #     if random.random() < self.epsilon:
    #         return random.randrange(self.OUTPUT_DIM)
    #     with torch.no_grad():
    #         state = state.to(self.device)
    #         q_values = self.policy_net(state)
    #         return q_values.argmax().item()

    def select_action(self, state):
        valid_actions = list(range(self.OUTPUT_DIM))
        
        # Exclude the opposite of the last action (if any)
        if self.last_action is not None:
            invalid = self.opposite_action[self.last_action]
            if invalid in valid_actions:
                valid_actions.remove(invalid)
    
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
    
                # Mask out the invalid action by setting Q-value to -inf
                q_values = q_values.squeeze()
                if self.last_action is not None:
                    q_values[self.opposite_action[self.last_action]] = float('-inf')
                action = q_values.argmax().item()
    
        self.last_action = action  # Update last action
        return action


    def store_transition(self, transition):
        self.memory.append(transition)

    def sample_batch(self):
        transitions = random.sample(self.memory, self.BATCH_SIZE)
        return zip(*transitions)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones.float()) * self.GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def training_loop(self):
        for episode in range(1, self.MAX_EPOCHS + 1):
            observation, info = self.env.reset()
            episode_return = 0
            done = False
            n_loop = 0

            while not done and n_loop < self.MAX_LOOP:
                state = torch.tensor(observation).float().unsqueeze(0).permute(0, 3, 1, 2)  # [B, C, H, W]
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state = torch.tensor(next_obs).float().unsqueeze(0).permute(0, 3, 1, 2)
                self.store_transition((state, action, reward, next_state, done))

                observation = next_obs
                episode_return += reward
                n_loop += 1

                self.optimize_model()

            self.episode_returns.append(episode_return)
            self.epsilon = max(self.EPSILON_END, self.epsilon * self.EPSILON_DECAY)

            if episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if episode % 10 == 0:
                avg_return = np.mean(self.episode_returns[-10:])
                print(f"Episode {episode}, Return: {episode_return}, AvgReturn(10): {avg_return:.2f}, Epsilon: {self.epsilon:.2f}")

            if np.mean(self.episode_returns[-10:]) >= self.REWARD_THRESHOLD:
                print(f"Solved in {episode} episodes.")
                break