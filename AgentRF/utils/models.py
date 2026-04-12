import json

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
import copy
import multiprocessing
import os
import tqdm
import time


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


import torch
import gc  # Import garbage collector

def sample_worker(model_state_dict, input_params):
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

    from WorldEnvironment.SnakeGame.SnakeEnv import envSnake
    env = envSnake()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(**input_params).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    observation, info = env.reset()
    reward_eps, obs_eps, val_eps = [], [], []
    done = False
    
    try:
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float, device=device)
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, value = model(obs_tensor)
                # Use .softmax and .sample
                dist = torch.distributions.Categorical(torch.nn.functional.softmax(action_logits, dim=-1))
                action = dist.sample()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            obs_eps.append(observation)
            reward_eps.append(reward)
            val_eps.append(value.item()) # .item() is important to move data to CPU
            
            observation = next_obs
            done = terminated or truncated

    finally:
        # --- GPU Memory Cleanup ---
        # 1. Delete large objects
        del model
        if 'obs_tensor' in locals(): del obs_tensor
        if 'action_logits' in locals(): del action_logits
        
        # 2. Clear Python garbage collector
        gc.collect()
        
        # 3. Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: Synchronize to ensure all kernels are finished
            torch.cuda.synchronize()

    return reward_eps, obs_eps, val_eps

# class PPOAgent:
#     def __init__(self):
#         env = envSnake()
#         self.device = "cuda" if torch.cuda.device_count() else "cpu"
#         obs_shape = env.observation_space
#         n_actions = math.prod(env.action_space)
#         env.close()

#         self.input_dim = obs_shape
#         self.share_hidden_dim = [8, 16, 32, 64]
#         self.kernel_size = [3, 3, 3, 3]
#         self.action_hidden_dim = 32
#         self.critic_hidden_dim = 32
#         self.action_dim = n_actions
#         self.critic_dim = 1

#         self.dropout = 0.0
#         self.learning_rate = 3e-4
#         self.discount = 0.99
#         self.GAE_parameter = 0.95
#         self.epsilon = 0.2
#         self.epochs = 10
#         self.batch_size = 64*8
#         self.iteration = 1000
#         self.num_actors = 5 #concurrent
#         self.num_episode = 100
#         self.num_episode_pre_actor = self.num_episode // self.num_actors
        
#         self.actor_critic = ActorCritic(self.input_dim, self.share_hidden_dim, self.kernel_size, self.action_hidden_dim, self.critic_hidden_dim, self.action_dim, self.critic_dim).to(device=self.device)
#         self.actor_critic_old = copy.deepcopy(self.actor_critic).to(device=self.device)

#         self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), self.learning_rate)

#         print(self.actor_critic)
#         print(obs_shape)

#     def running_sample(self):
#         reward_all = []
#         observation_all = []
#         value_all = []
#         for i in range(self.num_episode_pre_actor):
#             from WorldEnvironment.SnakeGame.SnakeEnv import envSnake
#             env = envSnake()

#             observation, info = env.reset()
#             reward_eps = []
#             observation_eps = []
#             value_eps = []
#             while truncated and terminated:
#                 action, value = self.actor_critic(observation.to(device=self.device))
#                 action_prob = F.softmax(action, dim = -1)
#                 dist = distributions.Categorical(action_prob)
#                 action = dist.sample()
#                 log_prob_action = dist.log_prob(action)
#                 # entropy = dist.entropy()
#                 observation_eps.append(observation)
#                 observation, reward, terminated, truncated, info = env.step(action.item())
#                 reward_eps.append(reward)
#                 value_eps.append(value)

#             reward_all.append(reward_eps)
#             observation_all.append(observation_eps)
#             value_all(value_eps)
#         return reward_all, observation_all, value_all
    
#     def advantage_estimates(self, reward, value):
#         # rev_reward = reversed(reward)
#         returns = []
#         for idx in range(len(reward)):
#             v = 0
#             for i in range(len(reward[idx:])):
#                 v += (self.discount**i) * (reward[i+idx])
#             returns.append(v)
#         return [r - v for r, v in zip(returns, value)], returns
    
#     def clipped_surrogate_objective(self, observation, advantage):
#         action, values = self.actor_critic(observation)
#         action_prob = F.softmax(action, dim = -1)
#         dist = distributions.Categorical(action_prob)
#         action = dist.sample()
#         log_prob_action = dist.log_prob(action)
#         prob_action = action_prob.gather(-1, action.unsqueeze(-1)).prod(dim=-1)  # Probability of the taken action

#         action_old, _ = self.actor_critic_old(observation)
#         action_prob_old = F.softmax(action_old, dim = -1)
#         dist_old = distributions.Categorical(action_prob_old)
#         action_old = dist_old.sample()
#         log_prob_action_old = dist_old.log_prob(action_old)
#         prob_action_old = action_prob_old.gather(-1, action_old.unsqueeze(-1)).prod(dim=-1)  # Probability of the taken action

#         ratio = prob_action/prob_action_old

#         surrogate = ratio*advantage
#         surrogate_clip = torch.clamp(ratio, min=1-self.epsilon, max=1+self.epsilon)*advantage

#         return torch.min(surrogate, surrogate_clip), values
    
#     def training(self):

#         model_params = {
#             "input_dim": self.input_dim,
#             "share_hidden_dim": self.share_hidden_dim,
#             "kernel_size": self.kernel_size,
#             "actor_hidden_dim": self.action_hidden_dim,
#             "critic_hidden_dim": self.critic_hidden_dim,
#             "actor_dim": self.action_dim,
#             "critic_dim": self.critic_dim
#         }

#         for it in range(self.iteration):
#             state_dict = self.actor_critic.cpu().state_dict()
            
#             # Create a list of arguments for each worker
#             worker_args = [(state_dict, model_params) for _ in range(self.num_episode)]

#             with multiprocessing.Pool(processes=self.num_actors) as pool:
#                 # Use starmap to pass multiple arguments to the worker
#                 results = pool.starmap(sample_worker, worker_args)
#             print("finish")
#             self.actor_critic.to(self.device)

#             # Flatten results
#             rewards = [r[0] for r in results]
#             observations = [r[1] for r in results]
#             values = [r[2] for r in results]

#             rewards_cpu = [r.cpu() if torch.is_tensor(r) else r for r in rewards]
#             values_cpu = [v.cpu() if torch.is_tensor(v) else v for v in values]

#             # rewards = [i for r in rewards for i in r]
#             # observations = [i for o in observations for i in o]
#             # values = [i for v in values for i in v]
            
#             with multiprocessing.Pool(processes=self.num_actors) as pool:
#                 args = [(r, v) for r, v in zip(rewards_cpu, values_cpu)]
#                 results = pool.starmap(self.advantage_estimates, args)

#             advantages = [torch.as_tensor(r[0]) for r in results]
#             returns = [torch.as_tensor(r[1]) for r in results]

#             # 1. Flatten the lists (ensure they are tensors)
#             flat_observations = [item.unsqueeze(0) for sublist in observations for item in sublist]

#             # 2. Cat first, THEN move to device
#             observations = torch.cat(flat_observations, dim=0).to(self.device).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
#             advantages = torch.cat(advantages, dim=0).to(self.device)
#             returns = torch.cat(returns, dim=0).to(self.device)

#             # สร้างแถบความก้าวหน้า (Progress Bar)
#             pbar = tqdm.tqdm(range(self.epochs))
            
#             for epoch in pbar:
#                 # สร้างตัวแปรเก็บค่าเฉลี่ยของ Loss ในแต่ละ Epoch
#                 epoch_loss_clip = []
#                 epoch_loss_value = []
            
#                 # Batch Training Loop
#                 for i in range(0, len(observations), self.batch_size):
#                     batch_obs = observations[i:i+self.batch_size]
#                     batch_adv = advantages[i:i+self.batch_size]
#                     batch_ret = returns[i:i+self.batch_size]
                    
#                     loss_clip, batch_values = self.clipped_surrogate_objective(batch_obs, batch_adv)
#                     values_loss = nn.MSELoss()(batch_values, batch_ret)
                    
#                     # คำนวณ Loss รวม (แนะนำให้ใช้ค่าสัมประสิทธิ์ 0.5 สำหรับ value loss ตามมาตรฐาน PPO)
#                     loss = -loss_clip.mean() + (0.5 * values_loss)
            
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
            
#                     # เก็บค่าเพื่อนำไปแสดงผล
#                     epoch_loss_clip.append(loss_clip.mean().item())
#                     epoch_loss_value.append(values_loss.item())
            
#                 # --- อัปเดต tqdm ตรงนี้ ---
#                 avg_clip = sum(epoch_loss_clip) / len(epoch_loss_clip)
#                 avg_val = sum(epoch_loss_value) / len(epoch_loss_value)
#                 avg_ret = torch.mean(returns).item()
            
#                 # ตั้งชื่อทางซ้าย (Iteration)
#                 pbar.set_description(f"Iteration: {it+1}/{self.iteration}")
                
#                 # ตั้งค่าตัวเลขทางขวา (Metrics)
#                 pbar.set_postfix({
#                     "Clip": f"{avg_clip:.4f}",
#                     "Val": f"{avg_val:.4f}",
#                     "Ret": f"{avg_ret:.2f}"
#                 })
            
#             self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())
#             # print(f"Iteration {it+1}/{self.iteration}, Loss Clip: {loss_clip.mean().item():.4f}, Value Loss: {values_loss.item():.4f}")
#             # print(f"Iteration {it+1}/{self.iteration}, Avg Return: {torch.mean(returns).item():.2f}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributions as distributions
# import copy
# import math
# import tqdm
# import numpy as np
# from torch.multiprocessing import Pool # Use torch's multiprocessing for CUDA
# from torch.utils.tensorboard import SummaryWriter

# class PPOAgent:
#     def __init__(self):
#         # env = envSnake() ... assuming initialized properly
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # --- Hyperparameters ---
#         self.learning_rate = 3e-4
#         self.discount = 0.99
#         self.gae_lambda = 0.95    # Actually used now
#         self.epsilon = 0.2        # Clip range
#         self.entropy_coef = 0.01  # Encourage exploration
#         self.vf_coef = 0.5        # Value function loss coefficient
#         self.max_grad_norm = 0.5  # Gradient clipping
        
#         self.epochs = 10
#         self.batch_size = 512*8
#         self.iteration = 100000
#         self.num_actors = 50
#         self.num_episode = 30000
#         self.num_episode_pre_actor = self.num_episode // self.num_actors

#         env = envSnake()
#         obs_shape = env.observation_space
#         n_actions = math.prod(env.action_space)
#         env.close()

#         self.input_dim = obs_shape
#         self.share_hidden_dim = [8, 16, 32, 64]
#         self.kernel_size = [3, 3, 3, 3]
#         self.action_hidden_dim = 32
#         self.critic_hidden_dim = 32
#         self.action_dim = n_actions
#         self.critic_dim = 1
        
#         # Initialize Actor-Critic (assuming you have this class)
#         self.actor_critic = ActorCritic2D(self.input_dim, self.share_hidden_dim, self.kernel_size, self.action_hidden_dim, self.critic_hidden_dim, self.action_dim, self.critic_dim).to(device=self.device)
#         # self.actor_critic = ActorCritic(self.input_dim, self.share_hidden_dim, self.action_hidden_dim, self.critic_hidden_dim, self.action_dim, self.critic_dim).to(device=self.device)
#         self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

#         # --- TENSORBOARD SETUP ---
#         # You can add a timestamp to the log_dir if you want to track multiple runs
#         import time
#         run_name = f"PPO_Snake_{int(time.time())}"
#         self.writer = SummaryWriter(log_dir=f"runs/{run_name}")
        
#     def compute_gae(self, rewards, values, dones):
#         """
#         Computes Generalized Advantage Estimation (GAE) in O(N) time.
#         Calculates backwards to save computation.
#         """
#         advantages = []
#         last_gae_lam = 0
        
#         # Append 0 to values to handle the edge case at the end of the trajectory
#         values = values + [0.0] 
        
#         for step in reversed(range(len(rewards))):
#             # If done is True, next_non_terminal is 0 (prevents bleeding values across episodes)
#             next_non_terminal = 1.0 - float(dones[step])
            
#             # delta = r + gamma * V(s') - V(s)
#             delta = rewards[step] + self.discount * values[step + 1] * next_non_terminal - values[step]
            
#             # A = delta + gamma * lambda * A'
#             last_gae_lam = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae_lam
#             advantages.insert(0, last_gae_lam)
            
#         returns = [adv + val for adv, val in zip(advantages, values[:-1])]
#         return advantages, returns

#     def running_sample(self): # (Or your sample_worker)
#         env = envSnake()
        
#         observations, actions, log_probs, rewards, values, dones, scores = [], [], [], [], [], [], []
        
#         observation, _ = env.reset()
        
#         # CRITICAL: No gradients needed during environment interaction
#         with torch.no_grad():
#             for step in range(self.num_episode_pre_actor): # Or a fixed max_steps
#                 obs_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
#                 # Forward pass
#                 action_logits, value = self.actor_critic(obs_tensor)
                
#                 # Sample action
#                 dist = distributions.Categorical(logits=action_logits) # Use logits directly
#                 action = dist.sample()
#                 log_prob = dist.log_prob(action)
                
#                 # Step environment
#                 next_observation, reward, terminated, truncated, score, _ = env.step(action.item())
#                 done = terminated or truncated
                
#                 # Store transition
#                 observations.append(observation)
#                 actions.append(action.item())
#                 log_probs.append(log_prob.item())
#                 rewards.append(reward)
#                 values.append(value.item())
#                 scores.append(score)
#                 dones.append(done)
                
#                 observation = next_observation
#                 if done:
#                     observation, _ = env.reset()

#         # Compute advantages for this specific worker's rollout
#         advantages, returns = self.compute_gae(rewards, values, dones)
        
#         return observations, actions, log_probs, advantages, returns, scores

#     def training(self):
#         # 1. (Optional) Setup Pool outside the loop to prevent memory leaks
#         # pool = multiprocessing.Pool(processes=self.num_actors)

#         for it in range(self.iteration):
#             # --- 1. COLLECT DATA ---
#             # ... Get results from workers here ...
#             # Assume we have flattened lists from all workers:
#             # all_obs, all_actions, all_old_log_probs, all_advantages, all_returns
#             all_obs = []
#             all_actions = []
#             all_old_log_probs = []
#             all_advantages = []
#             all_returns = []
#             all_scores = []

#             # pool_args = [(self.actor_critic.cpu().state_dict(), {
#             #     "input_dim": self.input_dim,
#             #     "share_hidden_dim": self.share_hidden_dim,
#             #     "kernel_size": self.kernel_size,
#             #     "actor_hidden_dim": self.action_hidden_dim,
#             #     "critic_hidden_dim": self.critic_hidden_dim,
#             #     "actor_dim": self.action_dim,
#             #     "critic_dim": self.critic_dim
#             # }) for _ in range(self.num_actors)]

#             # results = pool.starmap(self.running_sample, pool_args)

#             # for obs, actions, log_probs, advantages, returns in results:
#             #     all_obs.extend(obs)
#             #     all_actions.extend(actions)
#             #     all_old_log_probs.extend(log_probs)
#             #     all_advantages.extend(advantages)
#             #     all_returns.extend(returns)

#             for _ in range(self.num_actors):
#                 obs, actions, old_log_probs, advantages, returns, scores = self.running_sample()
#                 all_obs.extend(obs)
#                 all_actions.extend(actions)
#                 all_old_log_probs.extend(old_log_probs)
#                 all_advantages.extend(advantages)
#                 all_returns.extend(returns)
#                 all_scores.extend(scores)
            
#             # Convert to tensors
#             b_obs = torch.tensor(np.array(all_obs), dtype=torch.float32).to(self.device)
#             # Permute if your environment returns images: [B, H, W, C] -> [B, C, H, W]
#             b_obs = b_obs.permute(0, 3, 1, 2) 
            
#             b_actions = torch.tensor(all_actions, dtype=torch.long).to(self.device)
#             b_old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32).to(self.device)
#             b_advantages = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
#             b_returns = torch.tensor(all_returns, dtype=torch.float32).to(self.device)

#             # CRITICAL: Advantage Normalization (stabilizes training)
#             b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

#             dataset_size = len(b_obs)
#             indices = np.arange(dataset_size)
            
#             pbar = tqdm.tqdm(range(self.epochs))
#             for epoch in pbar:
#                 # CRITICAL: Shuffle data before slicing into mini-batches
#                 np.random.shuffle(indices)
                
#                 epoch_loss_clip = []
#                 epoch_loss_value = []
#                 epoch_entropy = []
                
#                 for start in range(0, dataset_size, self.batch_size):
#                     end = start + self.batch_size
#                     mb_inds = indices[start:end]
                    
#                     mb_obs = b_obs[mb_inds]
#                     mb_actions = b_actions[mb_inds]
#                     mb_old_log_probs = b_old_log_probs[mb_inds]
#                     mb_advantages = b_advantages[mb_inds]
#                     mb_returns = b_returns[mb_inds]
                    
#                     # --- 2. EVALUATE TAKEN ACTIONS ---
#                     action_logits, mb_values = self.actor_critic(mb_obs) # Permute if needed
#                     dist = distributions.Categorical(logits=action_logits)
                    
#                     # Get probabilities of the exact actions taken during the rollout
#                     new_log_probs = dist.log_prob(mb_actions)
#                     entropy = dist.entropy().mean() # Get entropy
                    
#                     # --- 3. CALCULATE LOSS ---
#                     # Ratio using exp subtraction (numerically stable)
#                     ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
#                     surr1 = ratio * mb_advantages
#                     surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                    
#                     actor_loss = -torch.min(surr1, surr2).mean()
                    
#                     # Value loss (squeeze to ensure shapes match)
#                     value_loss = nn.MSELoss()(mb_values.squeeze(-1), mb_returns)
                    
#                     # Total Loss = Policy + Value - Entropy
#                     loss = actor_loss + (self.vf_coef * value_loss) - (self.entropy_coef * entropy)
                    
#                     # --- 4. OPTIMIZE ---
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     # CRITICAL: Gradient clipping
#                     nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
#                     self.optimizer.step()
                    
#                     epoch_loss_clip.append(actor_loss.item())
#                     epoch_loss_value.append(value_loss.item())
#                     epoch_entropy.append(entropy.item())

#                 # Update tqdm
#                 pbar.set_description(f"Iteration: {it+1}/{self.iteration}")
#                 pbar.set_postfix({
#                     "Act Loss": f"{np.mean(epoch_loss_clip):.3f}",
#                     "Val Loss": f"{np.mean(epoch_loss_value):.3f}",
#                     "Ent": f"{np.mean(epoch_entropy):.3f}",
#                     "Ret": f"{b_returns.mean().item():.2f}",
#                     "Score": f"{np.mean(all_scores):.2f}",
#                     "Max Score": f"{np.max(all_scores):.2f}"
#                 })
            
#             # --- TENSORBOARD LOGGING ---
#             # Log metrics at the end of each iteration (not every epoch/minibatch) to keep graphs clean
#             self.writer.add_scalar("Loss/Actor", np.mean(epoch_loss_clip), it)
#             self.writer.add_scalar("Loss/Value", np.mean(epoch_loss_value), it)
#             self.writer.add_scalar("Loss/Entropy", np.mean(epoch_entropy), it)
            
#             self.writer.add_scalar("Performance/Average_Return", b_returns.mean().item(), it)
#             self.writer.add_scalar("Performance/Average_Score", np.mean(all_scores), it)
#             self.writer.add_scalar("Performance/Max_Score", np.max(all_scores), it)
            
#             # Optional: Log learning rate if you ever add a learning rate scheduler
#             # self.writer.add_scalar("Charts/Learning_Rate", self.optimizer.param_groups[0]['lr'], it)
            
#         # Close the writer when training is completely finished
#         self.writer.close()



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
import tqdm
import math
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

# --- 1. MULTIPROCESSING WORKER ---
def worker(remote, parent_remote, env_class, render_mode):
    """Runs a single environment in a separate CPU process."""
    parent_remote.close()
    env = env_class(render_mode=render_mode)
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                # Unpack your custom envSnake return signature
                next_obs, reward, term, trunc, score, _ = env.step(data)
                done = term or trunc
                
                # CRITICAL: Auto-reset the environment if it dies
                if done:
                    next_obs, _ = env.reset()
                    
                remote.send((next_obs, reward, done, score))
            elif cmd == 'reset':
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

class VecEnv:
    """Manages multiple environments concurrently."""
    def __init__(self, env_class, num_envs, render_mode=False):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [
            mp.Process(target=worker, args=(work_remote, remote, env_class, render_mode))
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]
        for p in self.processes:
            p.daemon = True # Ensure processes die if main script dies
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, scores = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(scores)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()


class PPOAgent:
    def __init__(self, config_path="config.json", config_name="default"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load the JSON configuration
        with open(config_path, 'r') as file:
            all_configs = json.load(file)
            
        if config_name not in all_configs:
            raise ValueError(f"Config '{config_name}' not found in {config_path}")
            
        config = all_configs[config_name]

        # 2. Assign Hyperparameters from config
        self.learning_rate = config["learning_rate"]
        self.discount = config["discount"]
        self.gae_lambda = config["gae_lambda"]
        self.epsilon = config["epsilon"]
        self.entropy_coef = config["entropy_coef"]
        self.vf_coef = config["vf_coef"]
        self.max_grad_norm = config["max_grad_norm"]

        # Network architecture
        self.share_hidden_dim = config["share_hidden_dim"]
        self.kernel_size = config["kernel_size"]
        self.action_hidden_dim = config["action_hidden_dim"]
        self.critic_hidden_dim = config["critic_hidden_dim"]

        # PPO Training parameters
        self.num_steps = config["num_steps"]
        self.batch_size = config["batch_size"]
        self.num_actors = config["num_actors"]
        self.epochs = config["epochs"]
        self.iteration = config["iteration"]
        self.warmup_iterations = config.get("warmup_iterations", int(self.iteration * 0.1))

        # 3. Dynamic Environment Variables (Kept in code)
        self.render_mode = config["render_mode"]  # e.g., "rgb_array" or "human"
        env = envSnake(render_mode=False)  # Initialize a temporary environment to get dimensions
        obs_shape = env.observation_space
        n_actions = math.prod(env.action_space)
        env.close()

        self.input_dim = obs_shape
        self.action_dim = n_actions
        self.critic_dim = 1
        
        # Initialize Actor-Critic (assuming you have this class)
        # self.actor_critic = ActorCritic2D(self.input_dim, self.share_hidden_dim, self.kernel_size, self.action_hidden_dim, self.critic_hidden_dim, self.action_dim, self.critic_dim).to(device=self.device)
        self.actor_critic = ActorCriticAttention2D(input_dim=self.input_dim, share_hidden_dim=self.share_hidden_dim, kernel_size=self.kernel_size, actor_hidden_dim=self.action_hidden_dim, critic_hidden_dim=self.critic_hidden_dim, actor_dim=self.action_dim, critic_dim=self.critic_dim).to(device=self.device)
        # self.actor_critic = ActorCritic(self.input_dim, self.share_hidden_dim, self.action_hidden_dim, self.critic_hidden_dim, self.action_dim, self.critic_dim).to(device=self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # 1. Initialize Concurrent Environments
        self.envs = VecEnv(envSnake, self.num_actors, self.render_mode)
        
        # 2. Tensorboard Writer
        import time
        run_name = f"PPO_Snake_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=f"runs/{run_name}")

        # Add a directory for checkpoints
        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = -float('inf')

    def update_learning_rate(self, current_iter):
        """Calculates and applies the learning rate based on warmup and linear decay."""
        if current_iter < self.warmup_iterations:
            # Linear Warmup: Ramp up from near 0 to base learning_rate
            lr_mult = (current_iter + 1) / max(1, self.warmup_iterations)
        else:
            # Linear Decay: Ramp down from base learning_rate to 0
            decay_total_iters = self.iteration - self.warmup_iterations
            elapsed_decay_iters = current_iter - self.warmup_iterations
            lr_mult = max(0.0, 1.0 - (elapsed_decay_iters / max(1, decay_total_iters)))
        
        current_lr = self.learning_rate * lr_mult
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
            
        return current_lr

    def save_checkpoint(self, iteration, filename="ppo_checkpoint.pth"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Create a dictionary containing everything needed to resume
        checkpoint = {
            'iteration': iteration,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score
        }
        
        torch.save(checkpoint, filepath)
        # Optional: print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filename="ppo_checkpoint.pth"):
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if os.path.isfile(filepath):
            print(f"Loading checkpoint '{filepath}'...")
            # Load to CPU first to avoid CUDA memory spikes, then move to device
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_score = checkpoint.get('best_score', -float('inf'))
            start_iteration = checkpoint.get('iteration', 0) + 1
            
            print(f"Loaded successfully. Resuming from iteration {start_iteration}.")
            return start_iteration
        else:
            print(f"No checkpoint found at '{filepath}'. Starting from scratch.")
            return 0

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        """Updated GAE to handle [Step, Actor] batched tensors."""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = values[t + 1]
                
            delta = rewards[t] + self.discount * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        return advantages, returns

    def training(self, checkpoint_filename="best_model.pth"):
        # --- OPTIONAL: Load previous checkpoint before starting ---
        try:
            start_it = self.load_checkpoint(checkpoint_filename)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            start_it = 0
        # start_it = 0 # Use this if starting fresh
        
        # Initial reset of all environments
        obs = self.envs.reset() # Shape: [num_actors, H, W, C]
        
        for it in range(start_it, self.iteration):
            current_lr = self.update_learning_rate(it)
            
            # --- 🌟 เพิ่มเงื่อนไขปรับขนาด num_steps กลางอากาศ 🌟 ---
            # ตัวอย่าง: เมื่อเทรนผ่านไปครึ่งทาง ให้เบิ้ล num_steps เป็น 2 เท่า (จาก 2048 -> 4096)
            if it == (self.iteration // 5):
                self.num_steps = 4096
                # ทางเลือกเพิ่มเติม: ถ้ากลัว PPO อัปเดตช้าลง อาจจะเพิ่ม batch_size ตามไปด้วย
                self.batch_size = 1024 
                print(f"\n🔥 [Phase 2] อัปเกรดความจำ! เพิ่ม num_steps เป็น {self.num_steps} และ Batch เป็น {self.batch_size} ในรอบที่ {it}\n")
            
            # (ถ้าอยากแบ่งเป็น 3 Phase ก็เพิ่มเงื่อนไข elif it == ... ได้เลย)
            # -----------------------------------------------------

            # --- 1. COLLECT DATA (CONCURRENTLY) ---
            # Pre-allocate memory on GPU for maximum speed
            # ตอนนี้ถ้า self.num_steps เปลี่ยน Buffer จะถูกจองพื้นที่กว้างขึ้นโดยอัตโนมัติ!
            b_obs = torch.zeros((self.num_steps, self.num_actors, self.input_dim[0], self.input_dim[1], self.input_dim[2]), dtype=torch.float32).to(self.device)
            b_actions = torch.zeros((self.num_steps, self.num_actors), dtype=torch.long).to(self.device)
            b_logprobs = torch.zeros((self.num_steps, self.num_actors), dtype=torch.float32).to(self.device)
            b_rewards = torch.zeros((self.num_steps, self.num_actors), dtype=torch.float32).to(self.device)
            b_dones = torch.zeros((self.num_steps, self.num_actors), dtype=torch.float32).to(self.device)
            b_values = torch.zeros((self.num_steps, self.num_actors), dtype=torch.float32).to(self.device)
            
            episode_scores = []

            for step in range(self.num_steps):
                # Convert obs to tensor, move to GPU, and permute for PyTorch image format (B, C, H, W)
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2) 
                
                with torch.no_grad():
                    # ONE single forward pass for all 50 actors!
                    action_logits, value = self.actor_critic(obs_tensor)
                    dist = distributions.Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                # Send actions back to CPU and step environments concurrently
                cpu_actions = action.cpu().numpy()
                next_obs, rewards, dones, scores = self.envs.step(cpu_actions)

                # Track scores for finished episodes
                for idx, d in enumerate(dones):
                    if d:
                        episode_scores.append(scores[idx])

                # Store data
                b_obs[step] = obs_tensor  # Keep in permuted format [B, C, H, W]
                b_actions[step] = action
                b_logprobs[step] = log_prob
                b_rewards[step] = torch.tensor(rewards, dtype=torch.float32).to(self.device)
                b_dones[step] = torch.tensor(dones, dtype=torch.float32).to(self.device)
                b_values[step] = value.squeeze(-1)

                obs = next_obs

            # --- 2. CALCULATE GAE ---
            with torch.no_grad():
                # Get value of the state *after* the final step for bootstrapping
                next_obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
                _, next_value = self.actor_critic(next_obs_tensor)
                next_value = next_value.squeeze(-1)
                next_done = torch.tensor(dones, dtype=torch.float32).to(self.device)
                
            advantages, returns = self.compute_gae(b_rewards, b_values, b_dones, next_value, next_done)

            # Flatten the batch to feed into the optimizer
            # [num_steps, num_actors, C, H, W] -> [num_steps * num_actors, C, H, W]
            b_obs_flat = b_obs.reshape((-1, *self.input_dim))
            b_actions_flat = b_actions.reshape(-1)
            b_logprobs_flat = b_logprobs.reshape(-1)
            b_advantages_flat = advantages.reshape(-1)
            b_returns_flat = returns.reshape(-1)
            
            # Normalize advantages
            b_advantages_flat = (b_advantages_flat - b_advantages_flat.mean()) / (b_advantages_flat.std() + 1e-8)

            # --- 3. OPTIMIZE ---
            dataset_size = len(b_obs_flat)
            indices = np.arange(dataset_size)
            
            pbar = tqdm.tqdm(range(self.epochs))
            for epoch in pbar:
                np.random.shuffle(indices)
                
                epoch_loss_clip, epoch_loss_value, epoch_entropy = [], [], []
                
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = indices[start:end]
                    
                    mb_obs = b_obs_flat[mb_inds]
                    mb_actions = b_actions_flat[mb_inds]
                    mb_old_log_probs = b_logprobs_flat[mb_inds]
                    mb_advantages = b_advantages_flat[mb_inds]
                    mb_returns = b_returns_flat[mb_inds]
                    
                    # Evaluate Taken Actions
                    action_logits, mb_values = self.actor_critic(mb_obs)
                    dist = distributions.Categorical(logits=action_logits)
                    
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    
                    # Loss Calculation
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(mb_values.squeeze(-1), mb_returns)
                    
                    loss = actor_loss + (self.vf_coef * value_loss) - (self.entropy_coef * entropy)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    epoch_loss_clip.append(actor_loss.item())
                    epoch_loss_value.append(value_loss.item())
                    epoch_entropy.append(entropy.item())

                # Update tqdm metrics
                avg_score = np.mean(episode_scores) if len(episode_scores) > 0 else 0
                max_score = np.max(episode_scores) if len(episode_scores) > 0 else 0
                
                pbar.set_description(f"Iteration: {it+1}/{self.iteration}")
                pbar.set_postfix({
                    "Act": f"{np.mean(epoch_loss_clip):.3f}",
                    "Val": f"{np.mean(epoch_loss_value):.3f}",
                    "Ent": f"{np.mean(epoch_entropy):.3f}",
                    "Score": f"{avg_score:.3f}",
                    "Max Score": f"{max_score:.1f}"
                })
            
            # After the optimization loop (pbar) finishes, calculate current metrics
            avg_score = np.mean(episode_scores) if len(episode_scores) > 0 else 0

            # --- 4. TENSORBOARD LOGGING ---
            self.writer.add_scalar("Loss/Actor", np.mean(epoch_loss_clip), it)
            self.writer.add_scalar("Loss/Value", np.mean(epoch_loss_value), it)
            self.writer.add_scalar("Loss/Entropy", np.mean(epoch_entropy), it)
            self.writer.add_scalar("Performance/Average_Return", b_returns_flat.mean().item(), it)
            if len(episode_scores) > 0:
                self.writer.add_scalar("Performance/Average_Score", avg_score, it)
                self.writer.add_scalar("Performance/Max_Score", max_score, it)
            self.writer.add_scalar("Hyperparameters/Learning_Rate", current_lr, it)

            # --- CHECKPOINT SAVING ---
            # 1. Save the best model
            if avg_score > self.best_score and len(episode_scores) > 0:
                self.best_score = avg_score
                self.save_checkpoint(it, filename=f"best_model.pth")
                print(f"*** New Best Score: {self.best_score:.2f}! Saved best_model.pth ***")
            
            # 2. Save a regular checkpoint every 100 iterations (adjust as needed)
            save_interval = 10
            if (it + 1) % save_interval == 0:
                self.save_checkpoint(it, filename=f"latest_model.pth") # Overwrites to save disk space
                # Or use filename=f"model_iter_{it+1}.pth" to keep ALL checkpoints

        self.envs.close()
        self.writer.close()
    
    def test(self, total_episodes=100, checkpoint_name="best_model.pth"):
        """
        Tests the agent across multiple concurrent environments.
        
        Args:
            total_episodes (int): How many total episodes to play across all actors.
            checkpoint_name (str): The name of the weights file to load from self.checkpoint_dir.
        """
        # 1. Load the best saved model
        self.load_checkpoint(checkpoint_name)
        
        # Set the network to evaluation mode
        self.actor_critic.eval()
        
        # 2. Reset the concurrent environments
        obs = self.envs.reset()
        
        completed_episodes = 0
        all_scores = []
        
        print(f"--- Starting Evaluation: {total_episodes} episodes using {self.num_actors} actors ---")
        
        # Disable gradient calculations for testing
        with torch.no_grad():
            while completed_episodes < total_episodes:
                # Prepare observation tensor [B, H, W, C] -> [B, C, H, W]
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                
                # Get action logits from the network
                action_logits, _ = self.actor_critic(obs_tensor)
                
                # --- EXPLOITATION: Choose the best action deterministically ---
                # Instead of dist.sample(), we use argmax to get the highest probability action
                actions = torch.argmax(action_logits, dim=-1)
                
                # Move to CPU and step environments
                cpu_actions = actions.cpu().numpy()
                next_obs, rewards, dones, scores = self.envs.step(cpu_actions)
                
                # Track completed episodes
                for idx, d in enumerate(dones):
                    if d:
                        completed_episodes += 1
                        all_scores.append(scores[idx])
                        print(f"Episode {completed_episodes}/{total_episodes} | Actor {idx} Score: {scores[idx]}")
                        
                        # Stop exactly when we hit the required number of episodes
                        if completed_episodes >= total_episodes:
                            break
                            
                obs = next_obs
                # time.sleep(0.1)

        # 3. Print Final Statistics
        avg_score = np.mean(all_scores)
        max_score = np.max(all_scores)
        min_score = np.min(all_scores)
        
        print("\n" + "="*30)
        print("      EVALUATION RESULTS      ")
        print("="*30)
        print(f"Total Episodes Played : {len(all_scores)}")
        print(f"Average Score         : {avg_score:.3f}")
        print(f"Maximum Score         : {max_score:.2f}")
        print(f"Minimum Score         : {min_score:.2f}")
        print("="*30)
        
        # Return to train mode just in case you want to continue training later
        self.actor_critic.train()
        
        return all_scores