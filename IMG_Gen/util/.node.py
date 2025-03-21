# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3  # pip install timm
from tqdm import tqdm  # pip install tqdm
import matplotlib.pyplot as plt  # pip install matplotlib
import torch.optim as optim
import numpy as np
from utils import YOLODataset_xml , postprocess
from PIL import Image

plt.switch_backend("Agg") # TKAgg


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]

# Residual Blocks


class ResBlock(nn.Module):
    def __init__(self, C: int, CO: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=CO)
        self.conv1 = nn.Conv2d(C, CO, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CO, CO, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.C = C
        self.CO = CO

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        if self.C == self.CO:
            return r + x
        else:
            return r


# class Attention(nn.Module):
#     def __init__(self, C: int, num_heads: int, dropout_prob: float):
#         super().__init__()
#         self.proj1 = nn.Linear(C, C*3)
#         self.proj2 = nn.Linear(C, C)
#         self.num_heads = num_heads
#         self.dropout_prob = dropout_prob

#     def forward(self, x):
#         h, w = x.shape[2:]
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.proj1(x)
#         x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
#         q, k, v = x[0], x[1], x[2]
#         x = F.scaled_dot_product_attention(
#             q, k, v, is_causal=False, dropout_p=self.dropout_prob)
#         x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
#         x = self.proj2(x)
#         return rearrange(x, 'b h w C -> b C h w')

class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.Q = nn.Linear(C, C)
        self.K = nn.Linear(C, C)
        self.V = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.softmax = nn.Softmax(dim=-1)
        # self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.proj = nn.Linear(C, C)

    def scaled_dot_product_attention(self, q, k, v):
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
        scores = self.softmax(scores)
        # scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def rearray(self, x):
        B, HxW, C = x.size()
        return x.view(B, HxW, self.num_heads, -1).transpose(1, 2) #(B, HxW, C) -> (B, h, HxW, C/h)

    def reverse_rearray(self, x):
        B, h, HxW, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, HxW, -1) #(B, h, HxW, C/h) -> (B, HxW, C)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).view(B,-1,C) # (B, (HxW), C)
        q = self.rearray(self.Q(x)) # (B, HxW, C)
        k = self.rearray(self.K(x)) # (B, HxW, C)
        v = self.rearray(self.V(x)) # (B, HxW, C)
        x = self.scaled_dot_product_attention(q, k, v)
        x = self.reverse_rearray(x) # (B, HxW, C)
        x = self.proj(x) # (B, HxW, C)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2) # (B, C, H, W)

        return x


class UnetLayer(nn.Module):
    def __init__(self,
                 upscale: bool,
                 attention: bool,
                 num_groups: int,
                 dropout_prob: float,
                 num_heads: int,
                 C: int):
        super().__init__()
        self.upscale = upscale
        if upscale == "same":
            self.ResBlock1 = ResBlock(
            C=C, CO=C//2, num_groups=num_groups, dropout_prob=dropout_prob)
            self.ResBlock2 = ResBlock(
            C=C//2, CO=C//2, num_groups=num_groups, dropout_prob=dropout_prob)
        elif upscale:
            self.ResBlock1 = ResBlock(
            C=C, CO=C//2, num_groups=num_groups, dropout_prob=dropout_prob)
            self.ResBlock2 = ResBlock(
            C=C//2, CO=C//2, num_groups=num_groups, dropout_prob=dropout_prob)
            self.conv = nn.ConvTranspose2d(
                C//2, C//4, kernel_size=4, stride=2, padding=1)
        elif upscale is None:
            self.ResBlock1 = ResBlock(
            C=C, CO=C, num_groups=num_groups, dropout_prob=dropout_prob)
            self.ResBlock2 = ResBlock(
            C=C, CO=C, num_groups=num_groups, dropout_prob=dropout_prob)
            self.conv = nn.ConvTranspose2d(
                C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.ResBlock1 = ResBlock(
            C=C, CO=C, num_groups=num_groups, dropout_prob=dropout_prob)
            self.ResBlock2 = ResBlock(
            C=C, CO=C, num_groups=num_groups, dropout_prob=dropout_prob)
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(
                C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        if self.upscale == "same":
            return x, x
        else:
            return self.conv(x), x


class UNET(nn.Module):
    def __init__(self,
                 Channels: List = [64, 128, 256, 512, 1024, 1024, 512, 256, 128],
                 Attentions: List = [False, False, False, False, True, False, False, False, False],
                 Upscales: List = [False, False, False, False, None, True, True, True, "same"],
                 num_groups: int = 32,
                 dropout_prob: float = 0.1,
                 num_heads: int = 8,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(
            input_channels, Channels[0], kernel_size=3, padding=1)
        # out_channels = (Channels[-1]//2)+Channels[0]
        # self.late_conv = nn.Conv2d(
            # out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(
            Channels[-1]//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(
            time_steps=time_steps, embed_dim=max(Channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        layer = getattr(self, f'Layer{self.num_layers//2 + 1}')
        x = layer(x, embeddings)[0]
        j = 1
        for i in range(self.num_layers//2 + 1, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((x, residuals[-j]), dim=1)
            x = layer(x, embeddings)[0]
            j += 1
            # x = torch.concat(
                # (layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(x))


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        self.beta = torch.linspace(
            1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(batch_size: int = 2,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr=2e-5,
          checkpoint_path: str = None,
          path_to_data: str = './data/CatVsDog'):
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    size = 16*2
    channel = 1

    # train_dataset = YOLODataset_xml(path=path_to_data, class_name=["cat", "dog"], width=size, height=size)
    transform = transforms.Compose([
            # Resize to the desired dimensions
            transforms.Resize((size, size)),
            # Convert PIL image or numpy array to a tensor
            transforms.ToTensor(),
            # transforms.Lambda(lambda x:x/255.0),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
                # 0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    # train_dataset = datasets.ImageFolder(root=path_to_data,transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET(input_channels=channel,output_channels=channel).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        # ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')
    
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x, _) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            t = torch.randint(0, num_time_steps, (batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (bidx):.5f}')

        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'ema': ema.state_dict()
        }
        torch.save(checkpoint, 'model/checkpoint/DDPM_T01.pth')
        model.eval()
        inference(model=model,size=size,channel=channel,epochs=i+1)
        model.train()


def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
    
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()


def inference(checkpoint_path: str = None,
              num_time_steps: int = 1000,
              ema_decay: float = 0.9999,
              model: UNET = None,
              size: int = None,
              channel: int = None,
              epochs: int = None):
    if model is None:
        checkpoint = torch.load(checkpoint_path)
        model = UNET().cuda()
        model.load_state_dict(checkpoint['weights'])
        # ema = ModelEmaV3(model, decay=ema_decay)
        # ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    with torch.no_grad():
        # model = ema.module.eval()
        for i in range(1):
            z = torch.randn(1, channel, size, size)
            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (
                    scheduler.beta[t]/((torch.sqrt(1-scheduler.alpha[t]))*(torch.sqrt(1-scheduler.beta[t]))))
                z = (
                    1/(torch.sqrt(1-scheduler.beta[t])))*z - (temp*model(z.cuda(), t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, channel, size, size)
                z = z + (e*torch.sqrt(scheduler.beta[t]))
                z = torch.clamp(z,-1,1)
            temp = scheduler.beta[0]/((torch.sqrt(1-scheduler.alpha[0]))
                                      * (torch.sqrt(1-scheduler.beta[0])))
            x = (1/(torch.sqrt(1-scheduler.beta[0]))) * \
                z - (temp*model(z.cuda(), [0]).cpu())

            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().view(size,size)
            # x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            # x = x.numpy() + float(x.min()*(-1))
            # x = ((x / float(x.max()))*255).astype(np.uint8)
            x = torch.clamp(x, -1, 1).numpy()
            x = (x + 1) / 2
            x = (x * 255).astype(np.uint8)
            Image.fromarray(x,mode="L").save("output/{}_{}.png".format(epochs,i))
            
            # x = torch.clamp(x, -1, 1).numpy()
            # x = (x + 1) / 2
            # x = (x * 255).astype(np.uint8)
            # Image.fromarray(x,mode="RGB").save("output/{}_{}.png".format(epochs,i))
            
            # plt.imsave("output/{}.png".format(i), x,cmap='gray')
            # plt.imshow(x)
            # plt.show()
            # display_reverse(images)
            images = []
