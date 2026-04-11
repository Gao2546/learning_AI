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
    

class ActorCritic2D(nn.Module):
    def __init__(self, input_dim: list[int], share_hidden_dim : list[int], kernel_size : list[int], actor_hidden_dim : int, critic_hidden_dim : int, actor_dim : int, critic_dim : int):
        super().__init__()
        self.silu = nn.SiLU(inplace= True)
        self.batch_norms = [nn.BatchNorm2d(dim) for dim in share_hidden_dim[1:]]
        self.input_layer = nn.Conv2d(input_dim[0], share_hidden_dim[0], kernel_size=3, stride=1, padding=1)
        self.learning_2d_feature = nn.Sequential(*(
                                    nn.Sequential( 
                                        nn.Conv2d(share_hidden_dim[idx], share_hidden_dim[idx+1], kernel_size=kernel_size[idx], stride=1),
                                        self.batch_norms[idx], 
                                        self.silu
                                    ) for idx in range(0, len(share_hidden_dim) - 1, 1)
                                ))
        self.hidden_2d_dim = [(input_dim[1] - sum(kernel_size[1:idx+1]) + idx) * (input_dim[2] - sum(kernel_size[1:idx+1]) + idx) * share_hidden_dim[idx]  for idx in range(len(kernel_size))]

        self.actor_layer = nn.Linear(self.hidden_2d_dim[-1], actor_hidden_dim)
        self.critic_layer = nn.Linear(self.hidden_2d_dim[-1], critic_hidden_dim)
        self.actor_output = nn.Linear(actor_hidden_dim, actor_dim)
        self.critic_output = nn.Linear(critic_hidden_dim, critic_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.learning_2d_feature(x)

        x = x.flatten(1)

        x_action = self.actor_layer(x)
        x_action = self.relu(x_action)
        x_action = self.actor_output(x_action)

        x_value = self.critic_layer(x)
        x_value = self.relu(x_value)
        x_value = self.critic_output(x_value)

        return x_action, x_value
    

import torch
import torch.nn as nn

class ActorCriticAttention2D(nn.Module):
    def __init__(self, 
                 input_dim: list[int], 
                 share_hidden_dim: list[int], 
                 kernel_size: list[int], 
                 actor_hidden_dim: int, 
                 critic_hidden_dim: int, 
                 actor_dim: int = 4,  # Defaulted to 4 for Snake Game (Up, Down, Left, Right)
                 critic_dim: int = 1, 
                 num_heads: int = 4,  # Number of attention heads
                 num_attention_layers: int = 2):
        super().__init__()
        
        self.silu = nn.SiLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        # 1. CNN Feature Extractor
        # NOTE: Using an nn.ModuleList or nn.Sequential properly ensures PyTorch tracks the weights
        layers = []
        in_channels = input_dim[0]
        
        # Build the Conv Block dynamically based on user lists
        for out_channels, k_size in zip(share_hidden_dim, kernel_size):
            # Using padding=k_size//2 keeps the spatial dimensions roughly the same, 
            # preventing the grid from shrinking to 0x0 on small Snake boards.
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=1, padding=k_size//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self.silu)
            in_channels = out_channels
            
        self.feature_extractor = nn.Sequential(*layers)

        # 2. Dynamic Shape Calculation (Safer than manual math)
        # We do a dummy forward pass to find out the exact dimensions after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            dummy_out = self.feature_extractor(dummy_input)
            _, c, h, w = dummy_out.shape
            
        self.embed_dim = c       # The number of channels becomes our embedding dimension
        self.seq_length = h * w  # The flattened width * height becomes our sequence length

        # 3. Positional Encoding
        # Attention has no concept of space/geometry. We MUST add positional embeddings 
        # so the snake knows where it is relative to the walls and the apple.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, self.embed_dim))

        # 4. Multi-Head Attention (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=self.embed_dim * 2,
            batch_first=True, # Expects inputs as (Batch, Sequence, Features)
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        # 5. Actor and Critic Heads
        # We will use Global Average Pooling after attention, so input is just `self.embed_dim`
        self.actor_layer = nn.Linear(self.embed_dim, actor_hidden_dim)
        self.actor_hidden = nn.Linear(actor_hidden_dim, actor_hidden_dim) # Optional extra hidden layer for the actor
        self.actor_output = nn.Linear(actor_hidden_dim, actor_dim) # Outputs 4 actions
        
        self.critic_layer = nn.Linear(self.embed_dim, critic_hidden_dim)
        self.critic_hidden = nn.Linear(critic_hidden_dim, critic_hidden_dim) # Optional extra hidden layer for the critic
        self.critic_output = nn.Linear(critic_hidden_dim, critic_dim)

    def forward(self, x):
        # 1. Extract visual features using CNN
        # Input shape: (Batch, Channels, Height, Width)
        x = self.feature_extractor(x) 
        
        B, C, H, W = x.shape

        # 2. Prepare for Attention
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        x = x.view(B, C, H * W)
        # Swap axes to match Transformer expected input: (Batch, Sequence_Length, Embed_Dim)
        x = x.permute(0, 2, 1)

        # 3. Add Positional Embeddings
        x = x + self.pos_embedding

        # 4. Multi-Head Self Attention
        x = self.transformer(x)

        # 5. Global Average Pooling
        # Average across the sequence length to get a single 1D vector per item in the batch
        # Shape goes from (B, Seq_Len, Embed_Dim) -> (B, Embed_Dim)
        x = x.mean(dim=1)

        # 6. Actor Head (Policy - 4 Actions)
        x_action = self.actor_layer(x)
        x_action = self.relu(x_action)
        x_action = self.actor_hidden(x_action) # Optional extra hidden layer
        x_action = self.relu(x_action)
        x_action = self.actor_output(x_action) # Raw logits for Up, Down, Left, Right

        # 7. Critic Head (Value)
        x_value = self.critic_layer(x)
        x_value = self.relu(x_value)
        x_value = self.critic_hidden(x_value) # Optional extra hidden layer
        x_value = self.relu(x_value)
        x_value = self.critic_output(x_value)

        return x_action, x_value


class ActorCritic(nn.Module):
    def __init__(self, input_dim: list[int], share_hidden_dim : list[int], actor_hidden_dim : int, critic_hidden_dim : int, actor_dim : int, critic_dim : int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.input_layer = nn.Linear(input_dim[0] * input_dim[1] * input_dim[2], share_hidden_dim[0])
        self.learning_flatten_feature = nn.Sequential(*(
                                    nn.Sequential( 
                                        nn.Linear(share_hidden_dim[idx], share_hidden_dim[idx+1]),
                                        nn.LayerNorm(share_hidden_dim[idx+1]), 
                                        self.relu
                                    ) for idx in range(0, len(share_hidden_dim) - 1, 1)
                                ))

        self.actor_layer = nn.Linear(share_hidden_dim[-1], actor_hidden_dim)
        self.critic_layer = nn.Linear(share_hidden_dim[-1], critic_hidden_dim)
        self.actor_output = nn.Linear(actor_hidden_dim, actor_dim)
        self.critic_output = nn.Linear(critic_hidden_dim, critic_dim)
    
    def forward(self, x):
        x = x.flatten(1)
        x = self.input_layer(x)
        x = self.learning_flatten_feature(x)

        x_action = self.actor_layer(x)
        x_action = self.relu(x_action)
        x_action = self.actor_output(x_action)

        x_value = self.critic_layer(x)
        x_value = self.relu(x_value)
        x_value = self.critic_output(x_value)

        return x_action, x_value