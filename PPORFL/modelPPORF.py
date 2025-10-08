# ppo_cartpole.py
import gym
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
SEED = 123
GAMMA = 0.99
LAM = 0.95                # GAE lambda
CLIP_EPS = 0.2
LR = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10       # how many epochs per update
MINI_BATCH_SIZE = 64
STEPS_PER_UPDATE = 2048  # collect this many steps then update
TOTAL_UPDATES = 2000     # outer loop iterations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utils / Seed ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        # common body
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # policy head
        self.policy = nn.Linear(hidden_size, act_dim)
        # value head
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp, value, dist.entropy()

    def get_logprob_value(self, obs, act):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy()
        return logp, value, entropy

# --- Rollout Buffer (on-policy) ---
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, obs, action, reward, done, logprob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(self, last_value, gamma=GAMMA, lam=LAM):
        # convert to numpy arrays
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1] 
        advantages = np.zeros_like(deltas)
        adv = 0.0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + gamma * lam * (1 - dones[t]) * adv
            advantages[t] = adv
        returns = advantages + values[:-1]
        # convert to tensors
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
        logprobs = torch.stack(self.logprobs).to(DEVICE).detach()
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return obs, actions, logprobs, values, returns, advantages

# --- PPO training function ---
def ppo_train():
    env = gym.make(ENV_NAME)
    # env.seed(SEED)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"Observation space: {obs_dim}, Action space: {act_dim}")

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer()
    episode_rewards = deque(maxlen=100)

    obs = env.reset()[0]
    ep_reward = 0
    total_steps = 0

    for update in range(1, TOTAL_UPDATES + 1):
        # collect rollouts
        # print(f"=== Update {update} ===")
        for step in range(STEPS_PER_UPDATE):
            # print("obs:", obs)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            # print("step:", step)
            # print(obs_tensor)
            # print(obs_tensor.shape)
            with torch.no_grad():
                logits, value = model.forward(obs_tensor)
                # print("logits:", logits)
                # print("value:", value)
                dist = Categorical(logits=logits)
                # print("dist:", dist)
                action = dist.sample().cpu().numpy()[0]
                # print("action:", action)
                logp = dist.log_prob(torch.tensor(action, device=DEVICE))
                # print("logp:", logp)
            next_obs, reward, done, info, _ = env.step(int(action))
            # print("next_obs:", next_obs)
            # print("reward:", reward)
            # print("done:", done)
            # print("info:", info)
            buffer.add(obs, action, reward, done, logp, value.item())
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            if done:
                # print(f"Episode finished after {step+1} steps.")
                episode_rewards.append(ep_reward)
                # print(f"Episode reward: {ep_reward}")
                obs = env.reset()[0]
                ep_reward = 0
        # print("Episode rewards (last 10):", list(episode_rewards)[-10:])
        # print(f"Collected {STEPS_PER_UPDATE} steps.")

        # compute last value for bootstrap
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, last_value = model.forward(obs_tensor)
            last_value = last_value.item()

        # prepare training data
        obs_b, actions_b, old_logprobs_b, values_b, returns_b, adv_b = buffer.compute_returns_and_advantages(last_value)
        buffer.clear()

        # PPO update: multiple epochs, minibatches
        dataset_size = obs_b.size(0)
        for epoch in range(UPDATE_EPOCHS):
            # generate permutation for minibatches
            perm = torch.randperm(dataset_size)
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                idx = perm[start:start + MINI_BATCH_SIZE]
                mb_obs = obs_b[idx]
                mb_actions = actions_b[idx]
                mb_old_logprobs = old_logprobs_b[idx]
                mb_returns = returns_b[idx]
                mb_adv = adv_b[idx]

                # current policy
                logits, values = model.forward(mb_obs)
                dist = Categorical(logits=logits)
                mb_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # ratio for clipping
                ratio = torch.exp(mb_logprobs - mb_old_logprobs)

                # clipped surrogate objective
                surrogate1 = ratio * mb_adv #
                surrogate2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # value loss (MSE)
                value_loss = (mb_returns - values).pow(2).mean()

                # total loss
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # Logging
        if len(episode_rewards) > 0:
            avg_reward = sum(episode_rewards) / len(episode_rewards)
        else:
            avg_reward = 0.0
        print(f"Update {update:4d}  Steps {total_steps:7d}  AvgReward(100) {avg_reward:6.2f}")

        # (optional) early stop if environment solved
        if avg_reward >= 475.0 and len(episode_rewards) >= 100:
            print("Environment solved!")
            break

    env.close()
    return model

if __name__ == "__main__":
    start_time = time.time()
    trained_model = ppo_train()
    print("Training done in {:.2f} sec".format(time.time() - start_time))


# ค่า reward รางวัลหากได้ผลลัพท์ที่ดี
# ค่า value บอกว่าถ้าเดินทางนั้น (action) ณ state นั้นๆ ในอนาคตจะดีแค่ไหน (ผลตอบแทนสะสม (reward) ในอนาคตหากเลือก action นั้นๆ ณ state นั้นๆ) ค่าที่คาดไว้
# ค่า advanctage อัตราการเปลี่ยนแปลงของ value บอกว่าเส้นทางที่เลือก มีผลกระทบมากแค่ไหน มีความก้าวหน้ามากแค่ไหน หลังจากที่เลือกแล้วได้ค่าที่ดีกว่าหรือแย่กว่าที่คาดไว้ (ค่า value) ค่า ลบ แย่กว่า ค่าบวก ดีกว่า
# ค่า return ค่า value ที่แท้จริง หลังจากคำนวณค่า advanctage แล้ว
# ค่า logprobs คือค่าความน่าจะเป็นของ (ความมั่นใจ) action นั้นๆใน state นั้นๆ ที่ model ทำนาย
# ค่า ratio ค่าอัตราความส่วนของค่า คาวมน่าจะเป็น (ความมั่นใจ) ในการเลือก action ของ model เก่า กับ model ใหม่ (new_model/old_model) (ที่กำลัังปรับ weight อยู่) อัตราส่วนของ logprobs ถ้ามีค่ามาก แสดงว่ามั่นใจกว่า model เก่ามาก
# ค่า surrogate1 = advanctage * ratio เราอยากเพิ่มค่านี้ให้เยอะๆ ถ้าค่า advanctage เป็นบวกแสดงว่าดี ต้องเพิ่ม ratio เยอะๆ(หมายความว่า ต้องปรับ model ให้ เพิ่มค่าความมั่นใจเยอๆ) ถ้าลบ แสดงว่าไม่ดี ต้องลดค่า ratio เยอะๆ
# ค่า surrogate2 = CLIP(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advanctage คล้ายๆกับ surrogate1 แต่ถ้ามันปรับเยอะเกินไปก็ให้ตัดออก
# ค่า policy loss:
### policy_loss = -torch.min(surrogate1, surrogate2).mean() เลือกเอาค่าที่ปรับน้อยที่สุด (ตัว ration ที่มีค่าน้อยที่สุด) (ใส่ลบ เพื่อเปลี่ยนจาก maximize เป็น minimize)
# ค่า value loss:
### value_loss = (mb_returns - values).pow(2).mean() ค่า value จริงๆ (คำนวณค่าจากอนาคตที่เดินทางไปจริงๆ) - ค่าที่ model ทำนายไว้
# ค่า entropy คือค่าความหลากหลายในการสุ่ม (ทุกๆ action มีความน่าจะเป็นเท่าๆกัน)
# total loss
### loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
### policy_loss ต้องการให้ค่านี้น้อยที่สุด (ค่านนี้จะติดลบ มากขึ้นเท่าใดก็ได้ การเรียนรู้ของ model ก็จะยิ่งเพิ่มขึ้นมาก ในตอนสุดท้ายหากค่านี้เยอะมากๆ จะทำให้ค่า loss อื่นๆมีความสำคัญลดลง)
### value_loss ต้องการให้ค่านี้เข้าใกล้ 0 (ถ้าค่านี้เข้าใกล้ 0 และจะส่งผลต่อการปรับค่า weight ของ model น้อยมาก)
### entropy ต้องการให้ค่านี้มากๆ เป็นตัวควบคุมความหลากหลายในการเลือก action (ทำให้เกิดการสุ่มเลือก action เกิดการลองใน หลายๆ action ไม่ติดอยู่กับ action เดิมๆ ในช่วงแรกๆ อิทธิพลของค่านี้จะลดลงไปเองเมื่อ model เรียนรู้ไปนานๆขึ้น) ซึ่งจะช่วยป้องกันไม่ให้ Agent ติดอยู่ใน Local Optimum


# แน่นอนครับ นี่คือคำอธิบายที่ปรับปรุงและขยายความเพิ่มเติมจากคอมเมนต์ของคุณ เพื่อให้แต่ละส่วนชัดเจนและถูกต้องยิ่งขึ้นครับ 👍

# ---

# ### **Reward (รางวัล)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Reward** คือ **สัญญาณตอบรับทันที** ที่ Agent ได้รับจากสิ่งแวดล้อม (Environment) หลังจากทำ Action หนึ่งๆ ใน State ณ เวลานั้น มันเป็นตัวเลขที่บอกว่า Action ที่เพิ่งทำไปนั้น "ดี" หรือ "ไม่ดี" ณ **วินาทีนั้น**

# * **ตัวอย่าง CartPole:**
#     * ได้ `+1` reward สำหรับทุกๆ step ที่ไม้ยังไม่ล้ม
#     * จบเกม (ไม้ล้ม) reward เป็น `0` และไม่ได้ไปต่อ

# Reward เป็นเหมือน "เงินเดือน" ที่ได้ทันที แต่ยังไม่ได้บอกว่าการตัดสินใจทำงานนี้จะส่งผลดีต่อ "ความมั่งคั่งในระยะยาว" หรือไม่

# ---

# ### **Value (ค่า Value)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Value** หรือ **State-Value Function ($V(s)$)** คือ **การประเมินผลตอบแทนรวมในอนาคต (Total Future Reward)** ที่คาดว่าจะได้รับ ถ้าเริ่มต้นจาก State นั้นๆ แล้วทำตาม Policy (นโยบายการตัดสินใจ) ปัจจุบันต่อไปจนจบเกม

# พูดง่ายๆ คือ มันตอบคำถามว่า "การอยู่ใน State นี้ มันดีแค่ไหนในระยะยาว?"

# * **เปรียบเทียบ:** ถ้า Reward คือ "เงินเดือน" ที่ได้ทันที Value ก็เปรียบเสมือน "ศักยภาพในการสร้างรายได้ในอนาคต" จากตำแหน่งงานปัจจุบัน
# * **ในโค้ด:** `self.value` head ของโมเดลพยายามเรียนรู้ที่จะทำนายค่านี้ มันคือค่าที่ "คาดไว้" (Expected Value) ว่าอนาคตจะเป็นอย่างไรจากจุดนี้

# ---

# ### **Advantage (ค่า Advantage)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Advantage Function ($A(s,a)$)** เป็นตัวชี้วัดว่า Action หนึ่งๆ ที่เลือกทำใน State นั้นๆ **ดีกว่าค่าเฉลี่ย** ที่คาดไว้จาก State นั้นมากแค่ไหน

# สูตรแนวคิดคือ: $A(s, a) = Q(s, a) - V(s)$

# * $Q(s, a)$: คือค่า Value ที่คาดว่าจะได้รับถ้าเลือก Action `a` ใน State `s`
# * $V(s)$: คือค่า Value เฉลี่ยของ State `s` (จากการทำทุก Action ที่เป็นไปได้ตาม Policy)

# **อธิบายง่ายๆ:**
# * **Advantage > 0 (ค่าบวก):** Action ที่เราทำนั้นให้ผลลัพธ์ดีกว่าที่คาดไว้ (เป็น "Surprise" ที่ดี) ควรทำ Action นี้บ่อยขึ้น
# * **Advantage < 0 (ค่าลบ):** Action ที่เราทำนั้นให้ผลลัพธ์แย่กว่าที่คาดไว้ (เป็น "Surprise" ที่น่าผิดหวัง) ควรทำ Action นี้น้อยลง

# ในโค้ดนี้ใช้เทคนิคที่เรียกว่า **Generalized Advantage Estimation (GAE)** ซึ่งเป็นวิธีคำนวณ Advantage ที่ซับซ้อนขึ้นเล็กน้อย เพื่อลดความคลาดเคลื่อนและทำให้การเรียนรู้เสถียรขึ้น

# ---

# ### **Return (ค่า Return)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Return ($G_t$)** คือ **ผลรวมของ Reward ที่เกิดขึ้นจริง** นับตั้งแต่เวลา $t$ ไปจนจบ Episode (มีการคิดลดทอนค่าตามเวลาด้วย `gamma`) มันคือ "ผลลัพธ์จริงๆ" ที่เกิดขึ้น

# **ลำดับที่ถูกต้องคือ:**
# 1.  Agent เล่นไปจนจบ Episode หรือเก็บข้อมูลครบ `STEPS_PER_UPDATE`
# 2.  เรามองย้อนกลับไป และคำนวณ **Return ($G_t$)** ที่เกิดขึ้นจริงในแต่ละ Step
# 3.  เราใช้ Return ($G_t$) และ Value ที่โมเดลทำนายไว้ ($V(s)$) มาคำนวณ **Advantage**

# ดังนั้น **Return** คือเป้าหมาย (Target) ที่เราจะใช้สอน Value Function ให้ทำนายได้แม่นยำขึ้น ไม่ใช่ผลลัพธ์ที่ได้หลังจากการคำนวณ Advantage ครับ

# ในโค้ดนี้ `returns = advantages + values[:-1]` คือการสร้าง "Target" สำหรับการอัปเดต Value Function โดยใช้ค่า Advantage ที่คำนวณจาก GAE ซึ่งเป็นเทคนิคที่เสถียรกว่าการใช้ Return ดิบๆ

# ---

# ### **Logprobs (ค่า Log Probs)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Logprobs** คือ **ค่า Logarithm ของความน่าจะเป็น (Probability)** ที่ Policy จะเลือก Action หนึ่งๆ ใน State นั้นๆ

# * **ทำไมต้องใช้ Log?**
#     1.  **ความเสถียรทางตัวเลข (Numerical Stability):** ความน่าจะเป็นเป็นตัวเลขระหว่าง 0-1 การคูณเลขทศนิยมเล็กๆ จำนวนมากเข้าด้วยกันอาจทำให้ค่าเข้าใกล้ 0 จนเกิดปัญหา (underflow) การเปลี่ยนเป็น Log จะทำให้การคูณกลายเป็นการบวก ซึ่งจัดการได้ง่ายกว่า
#     2.  **ความสะดวกในการคำนวณ:** ในการทำ Optimization การหาอนุพันธ์ (derivative) ของผลบวกนั้นง่ายกว่าผลคูณ

# ค่า Logprobs จึงเป็นตัวแทน "ความมั่นใจ" ของโมเดลในการเลือก Action นั้นๆ ในรูปแบบ Log scale

# ---

# ### **Ratio (ค่า Ratio)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Ratio** คือ **อัตราส่วนความน่าจะเป็น** ระหว่าง Policy **ใหม่** (ที่กำลังจะอัปเดต) กับ Policy **เก่า** (ที่ใช้เก็บข้อมูล) ในการเลือก Action เดียวกัน

# $$\text{ratio} = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)}$$

# * **Ratio > 1:** Policy ใหม่มีความมั่นใจที่จะเลือก Action นี้ **มากกว่า** Policy เก่า
# * **Ratio < 1:** Policy ใหม่มีความมั่นใจที่จะเลือก Action นี้ **น้อยกว่า** Policy เก่า
# * **ในโค้ด:** คำนวณจาก `torch.exp(mb_logprobs - mb_old_logprobs)` ซึ่งเท่ากับสมการข้างบน เพราะ $e^{(\log(a) - \log(b))} = e^{\log(a/b)} = a/b$

# Ratio เป็นหัวใจของ PPO ที่บอกเราว่าการอัปเดตครั้งนี้จะเปลี่ยน Policy ไปมากน้อยแค่ไหน

# ---

# ### **Surrogate Objectives และ Policy Loss**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# เป้าหมายหลักของเราคือการทำให้ผลตอบแทนสูงขึ้น โดยการปรับ Policy ไปในทิศทางที่ให้ Advantage เป็นบวก

# * `surrogate1 = ratio * mb_adv`: นี่คือเป้าหมายพื้นฐาน ถ้า `mb_adv` เป็นบวก (ดี) เราก็อยากเพิ่ม `ratio` (ทำ Action นี้บ่อยขึ้น) ถ้า `mb_adv` เป็นลบ (แย่) เราก็อยากลด `ratio` (ทำ Action นี้น้อยลง) **แต่...** การปรับ `ratio` มากเกินไปอาจทำให้การเรียนรู้พังได้!

# * `surrogate2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv`: นี่คือเวอร์ชัน "ปลอดภัย" ของเป้าหมาย มันคือการ **Clip** หรือ **จำกัดขอบเขตของ `ratio`** ไม่ให้เปลี่ยนแปลงไปจากเดิม (คือ 1.0) มากเกินกว่าค่า `CLIP_EPS` (เช่น 0.2) เพื่อป้องกันการอัปเดตที่ก้าวกระโดดเกินไป

# * `policy_loss = -torch.min(surrogate1, surrogate2).mean()`: PPO เลือกใช้ค่าที่ **น้อยกว่า** ระหว่าง `surrogate1` และ `surrogate2` เสมอ นี่คือหลักการ "มองโลกในแง่ร้าย" (Pessimistic) เพื่อให้การอัปเดตเป็นไปอย่างระมัดระวังที่สุด และการใส่เครื่องหมาย **ลบ (`-`)** ก็เพื่อเปลี่ยนปัญหาจากการ Maximization (ทำให้รางวัลสูงสุด) ไปเป็น Minimization (ทำให้ Loss ต่ำสุด) ซึ่งเป็นสิ่งที่ Optimizer ทั่วไปทำ



# ---

# ### **Value Loss**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# `value_loss = (mb_returns - values).pow(2).mean()`: ส่วนนี้ตรงไปตรงมา คือการวัดว่า Value ที่โมเดลทำนาย (`values`) แตกต่างจาก "เป้าหมาย" ที่เราคำนวณไว้ (`mb_returns`) มากแค่ไหน (วัดด้วย Mean Squared Error) เป้าหมายคือทำให้ Value Loss เข้าใกล้ 0 ที่สุด ซึ่งหมายความว่า "นักวิจารณ์" (Critic) ของเราทำนายอนาคตได้แม่นยำขึ้นเรื่อยๆ

# ---

# ### **Entropy (ค่า Entropy)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# **Entropy** คือ **ค่าวัดความไม่แน่นอน** หรือความ "สุ่ม" ของ Policy
# * **Entropy สูง:** Policy มีความลังเล กระจายความน่าจะเป็นไปให้หลายๆ Action (คล้ายๆ กัน) ซึ่งดีต่อการ **สำรวจ (Exploration)** ในช่วงแรกๆ
# * **Entropy ต่ำ:** Policy มีความมั่นใจสูง ความน่าจะเป็นจะเทไปที่ Action ใด Action หนึ่งอย่างชัดเจน (ดีต่อการ **ใช้ประโยชน์ (Exploitation)** เมื่อเรียนรู้ไปสักพักแล้ว)

# ---

# ### **Total Loss (Loss รวม)**

# **คำอธิบายที่ปรับปรุงแล้ว:**
# `loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy`
# Loss รวมคือการนำเป้าหมายทั้ง 3 ส่วนมารวมกัน เพื่อหาจุดสมดุลที่ดีที่สุดในการอัปเดตโมเดล:

# 1.  **`policy_loss` (เป้าหมายหลัก):** ปรับปรุง Policy ให้ดีขึ้นอย่างระมัดระวัง
# 2.  **`VALUE_COEF * value_loss` (เป้าหมายรอง):** ปรับปรุงความแม่นยำของตัวประเมินค่า (Critic)
# 3.  **`- ENTROPY_COEF * entropy` (ตัวช่วย):** ส่งเสริมการสำรวจ (Exploration) โดยการ "ลงโทษ" Policy ที่มั่นใจในตัวเองเร็วเกินไป (Entropy ต่ำ) เราใส่เครื่องหมายลบ เพราะเราต้องการ **Maximize** Entropy แต่ Optimizer ทำได้แค่ **Minimize** Loss ดังนั้นการ Minimize "ลบ Entropy" จึงเท่ากับการ Maximize Entropy นั่นเอง

# `VALUE_COEF` และ `ENTROPY_COEF` คือค่าน้ำหนักที่บอกว่าเราให้ความสำคัญกับแต่ละเป้าหมายมากน้อยแค่ไหน