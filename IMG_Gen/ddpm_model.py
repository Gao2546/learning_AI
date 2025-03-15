import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random
import numpy as np
import signal
import sys
from torchvision import transforms, datasets

from util.node import *
from util.utils import *

# Device setup
device = torch.device("cuda")
world_size = torch.cuda.device_count()  # Number of GPUs

# Function to handle DDP setup
def ddp_setup(rank: int, world_size: int):
    """Setup Distributed Data Parallel (DDP) environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Function to set random seed
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Function to handle keyboard interrupt
def signal_handler(sig, frame):
    print("Training interrupted by user")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sys.exit(0)

# Function to be run by mp.spawn()
def train_ddp(rank, world_size, train_dataset, batch_size, model_ckp, model_VQVAE):
    ddp_setup(rank, world_size)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=4)

    # 🟢 FIX: Move Model to Rank-Specific Device & Wrap with DDP
    model = VQVAETrainer(in_c=3, 
                               out_c=3, 
                               down_sampling_times=2, 
                               encode_laten_channel=4, 
                               Z_size=16384, 
                               load_model_path=model_ckp, 
                               lr=1e-3).to(rank)
    model = diffusion_model(
        in_c=3, 
        out_c=3, 
        st_channel=64, 
        channel_multi=[1, 2, 4], 
        att_channel=64, 
        embedding_time_dim=64, 
        time_exp=256, 
        num_head=1, 
        d_model=32, 
        num_resbox=2, 
        allow_att=[True, True, True], 
        concat_up_down=True, 
        concat_all_resbox=True, 
        down_sampling_times=2, 
        encode_laten_channel=4, 
        Z_size=16384, 
        load_model_path=model_ckp, 
        load_model_path_VQVAE=model_VQVAE, 
        lr=1e-4
    )

    # model = diffusion_model_No_VQVAE(
    #     in_c=3, 
    #     out_c=3, 
    #     st_channel=64, 
    #     channel_multi=[1, 2, 4], 
    #     att_channel=64, 
    #     embedding_time_dim=64, 
    #     time_exp=256, 
    #     num_head=4, 
    #     d_model=32, 
    #     num_resbox=2, 
    #     allow_att=[True, True, True], 
    #     concat_up_down=True, 
    #     concat_all_resbox=True, 
    #     load_model_path=model_ckp
    # )
    # Count model parameters
    model_size = sum(p.numel() for p in model.vqvae.parameters() if p.requires_grad)
    print(f"Model size: {model_size} trainable parameters")
    model = DDP(model, device_ids=[rank])

    print(f"Rank {rank}: Model loaded. Starting training...")
    model.module.train_model(train_loader, num_epochs=100)

    dist.destroy_process_group()

def main():
    """Main function that initializes DDP and starts training."""
    signal.signal(signal.SIGINT, signal_handler)

    # Set seed
    seed = 42
    set_seed(seed)

    # Paths
    model_ckp = None#"model/checkpoint/DDPM_T_VQVAE4.pth"
    model_VQVAE_path = "model/checkpoint/VQVAE1.pth"
    path_to_data = "./data/104Flower_resized"

    # Training setup
    size = 16 * 8
    batch_size = 16 * 32

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load dataset
    train_dataset = datasets.ImageFolder(root=path_to_data, transform=transform)
    
    print("Data loaded successfully!")

    print("Starting distributed training...")

    # Use mp.spawn() to run training across multiple GPUs
    #mp.spawn(train_ddp, args=(world_size, model_VQVAE, train_dataset, batch_size), nprocs=world_size, join=True)
    mp.spawn(train_ddp, args=(world_size, train_dataset, batch_size, model_ckp, model_VQVAE_path), nprocs=world_size, join=True)

    print("Training completed!")

if __name__ == "__main__":
    main()
