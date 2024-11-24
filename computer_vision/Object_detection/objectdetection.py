import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.node import *

# Sample Dataset class


class YOLODataset(Dataset):
    def __init__(self, images, targets):
        """
        :param images: List of image tensors (C, H, W).
        :param targets: List of target tensors (bounding boxes + classes).
        """
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


# Dummy data (replace with real images and annotations)
images = [torch.randn(3, 640//1, 640//1) for _ in range(10)]  # 100 random images
targets = [[torch.randn(3, sizess, sizess, (2+4)*3)
           for _ in range(10)] for sizess in [80,40,20]]  # Example target tensors

# Dataset and DataLoader
dataset = YOLODataset(images, [targets[0], targets[1], targets[2]])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model, Loss, Optimizer
num_classes = 2
model = YOLOv8(num_classes=num_classes)
criterion = YOLOLoss()
optimizer = setup_optimizer(model=model)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (inputs, targetss) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        # print(targets.size())

        # Forward pass
        outputs = model(inputs)
        # print(outputs[0].size())
        # print(outputs[1].size())

        # Compute loss
        loss = 0
        for output,target in zip(outputs, targetss):  # Combine losses from all detection heads
            loss += criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss
        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx +
              1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{
          epoch+1}/{num_epochs}] Completed. Average Loss: {epoch_loss / len(dataloader):.4f}")

print("Training Complete.")
