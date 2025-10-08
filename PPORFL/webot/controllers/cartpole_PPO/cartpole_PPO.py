from controller import Robot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import time

# --- Hyperparameters ---

GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
MINI_BATCH_SIZE = 64
STEPS_PER_UPDATE = 1024
TOTAL_UPDATES = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO Actor-Critic ---

class ActorCritic(nn.Module):
def **init**(self, obs_dim, act_dim, hidden=64):
super().**init**()
self.shared = nn.Sequential(
nn.Linear(obs_dim, hidden), nn.Tanh(),
nn.Linear(hidden, hidden), nn.Tanh()
)
self.policy = nn.Linear(hidden, act_dim)
self.value = nn.Linear(hidden, 1)

```
def forward(self, x):
    x = self.shared(x)
    return self.policy(x), self.value(x).squeeze(-1)
```

# --- Rollout Buffer ---

class RolloutBuffer:
def **init**(self):
self.obs, self.actions, self.rewards, self.dones, self.logprobs, self.values = [], [], [], [], [], []
def add(self, *args):
self.obs.append(args[0]); self.actions.append(args[1]); self.rewards.append(args[2])
self.dones.append(args[3]); self.logprobs.append(args[4]); self.values.append(args[5])
def clear(self): self.**init**()
def compute(self, last_value, gamma=GAMMA, lam=LAM):
rewards = np.array(self.rewards, dtype=np.float32)
values = np.array(self.values + [last_value], dtype=np.float32)
dones = np.array(self.dones, dtype=np.float32)
deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
advs = np.zeros_like(deltas); adv = 0.0
for t in reversed(range(len(deltas))):
adv = deltas[t] + gamma * lam * (1 - dones[t]) * adv
advs[t] = adv
returns = advs + values[:-1]
obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=DEVICE)
actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
logprobs = torch.stack(self.logprobs).detach().to(DEVICE)
values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)
advs = torch.tensor(advs, dtype=torch.float32, device=DEVICE)
returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
advs = (advs - advs.mean()) / (advs.std() + 1e-8)
return obs, actions, logprobs, values, returns, advs

# --- Webots Environment Wrapper ---

class WebotsEnv:
def **init**(self):
self.robot = Robot()
self.timestep = int(self.robot.getBasicTimeStep())
self.left_motor = self.robot.getMotor("left wheel motor")
self.right_motor = self.robot.getMotor("right wheel motor")
self.left_motor.setPosition(float("inf"))
self.right_motor.setPosition(float("inf"))
self.left_motor.setVelocity(0.0)
self.right_motor.setVelocity(0.0)
# example sensors
self.sensors = [self.robot.getDistanceSensor(f"ds{i}") for i in range(2)]
for s in self.sensors: s.enable(self.timestep)

```
    self.max_steps = 1000
    self.steps = 0

def reset(self):
    self.steps = 0
    self.robot.simulationReset()
    obs = self._get_obs()
    return obs

def step(self, action):
    # Example: 2 actions = [forward, turn]
    if action == 0:
        self.left_motor.setVelocity(3.0)
        self.right_motor.setVelocity(3.0)
    elif action == 1:
        self.left_motor.setVelocity(-2.0)
        self.right_motor.setVelocity(2.0)

    self.robot.step(self.timestep)
    self.steps += 1

    obs = self._get_obs()
    reward = self._compute_reward(obs)
    done = self.steps >= self.max_steps
    return obs, reward, done, {}

def _get_obs(self):
    return np.array([s.getValue() for s in self.sensors], dtype=np.float32)

def _compute_reward(self, obs):
    # Example reward: encourage moving forward without hitting obstacle
    return -obs.mean() * 0.001 + 1.0
```

# --- PPO Train Loop ---

def ppo_train():
env = WebotsEnv()
obs_dim = len(env._get_obs())
act_dim = 2   # forward or turn

```
model = ActorCritic(obs_dim, act_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
buffer = RolloutBuffer()
episode_rewards = deque(maxlen=100)

obs = env.reset()
ep_reward = 0
total_steps = 0

for update in range(1, TOTAL_UPDATES+1):
    for step in range(STEPS_PER_UPDATE):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        next_obs, reward, done, _ = env.step(action.item())
        buffer.add(obs, action.item(), reward, done, logp, value.item())
        obs = next_obs
        ep_reward += reward
        total_steps += 1
        if done:
            episode_rewards.append(ep_reward)
            obs = env.reset()
            ep_reward = 0

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        _, last_value = model(obs_tensor)
        last_value = last_value.item()

    obs_b, actions_b, old_logprobs_b, values_b, returns_b, adv_b = buffer.compute(last_value)
    buffer.clear()

    dataset_size = obs_b.size(0)
    for epoch in range(UPDATE_EPOCHS):
        perm = torch.randperm(dataset_size)
        for start in range(0, dataset_size, MINI_BATCH_SIZE):
            idx = perm[start:start+MINI_BATCH_SIZE]
            mb_obs, mb_actions, mb_old_logprobs = obs_b[idx], actions_b[idx], old_logprobs_b[idx]
            mb_returns, mb_adv = returns_b[idx], adv_b[idx]

            logits, values = model(mb_obs)
            dist = Categorical(logits=logits)
            mb_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(mb_logprobs - mb_old_logprobs)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0-CLIP_EPS, 1.0+CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (mb_returns - values).pow(2).mean()
            loss = policy_loss + VALUE_COEF*value_loss - ENTROPY_COEF*entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    avg_reward = sum(episode_rewards)/len(episode_rewards) if episode_rewards else 0
    print(f"Update {update:4d}, Steps {total_steps:6d}, AvgReward(100) {avg_reward:.2f}")
```

if **name** == "**main**":
start = time.time()
ppo_train()
print("Training finished in", time.time()-start, "sec")
