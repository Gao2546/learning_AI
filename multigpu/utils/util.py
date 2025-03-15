import torch
import torch.nn.functional as F
from utils import MyTrainDataset
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# Custom dataset
class RandomDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def ddp_setup(rank: int, world_size: int):
    """Setup Distributed Data Parallel (DDP) environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train(rank: int, world_size: int):
    """Train the model using Distributed Data Parallel (DDP)."""
    ddp_setup(rank, world_size)

    # Load dataset
    dataset = RandomDataset(1000)  # 1000 samples
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=2)

    # Define model, loss function, and optimizer
    model = nn.Linear(10, 1).cuda(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)  # Ensure shuffling works correctly
        total_loss = 0

        for batch in dataloader:
            x, y = batch
            x, y = x.cuda(rank), y.cuda(rank)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
