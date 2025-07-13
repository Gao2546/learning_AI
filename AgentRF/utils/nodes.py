import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return x
    
# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout):
#         super().__init__()

#         self.layer1 = nn.Linear(input_dim, input_dim//2)
#         self.layer2 = nn.Linear(input_dim//2, input_dim//4)
#         self.layer3 = nn.Linear(input_dim//4, input_dim//8)
#         self.layer4 = nn.Linear(input_dim//8, input_dim//16)
#         self.layer5 = nn.Linear(input_dim//16, input_dim//32)
#         self.layer6 = nn.Linear(input_dim//32, input_dim//64)
#         self.layer7 = nn.Linear(input_dim//64, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer2(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer3(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer4(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer5(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer6(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         x = self.layer7(x)
#         x = self.dropout(x)
#         x = F.relu(x)

#         return x
    
class PolicyNetworkCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()

        self.layer1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Linear(32*20*30, 8)
        self.layer4 = nn.Linear(8, 4)
        self.Norm1 = nn.BatchNorm2d(16)
        self.Norm2 = nn.LayerNorm(8)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.Norm1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = x.flatten(1)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.Norm2(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        return x
    
class PolicyNet(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(PolicyNet, self).__init__()
        H, W, C = obs_shape  # You can transpose the input from (H, W, C) to (C, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * H * W, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) → (1, C, H, W)
        x = self.conv(x)
        x = self.fc(x)
        return F.softmax(x, dim=-1)
    

class DQNCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_dim[1] * input_dim[2], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.cnn(x)