import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return normalized_returns


def calculate_loss(stepwise_returns, log_prob_actions):
    # print(stepwise_returns)
    # print("--------------------------------")
    # print(log_prob_actions)
    # print("==========================================")
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss

