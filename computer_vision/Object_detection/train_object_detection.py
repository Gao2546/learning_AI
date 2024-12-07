import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.node import *
import os
from utils.loaddata import YOLODataset_xml


path_data = "/home/athip/psu/learning_AI/computer_vision/Object_detection/images/CatVsDog"
dataset = YOLODataset_xml(path=path_data, class_name=["cat", "dog"], width=640, height=640)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, Loss, Optimizer
num_classes = 2
model = YOLOv8(num_classes=num_classes)
criterion = YOLOLoss()
optimizer = setup_optimizer(model=model)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (inputs, (targets80, targets40, targets20), real_size) in enumerate(dataloader):
        inputs = inputs.to(device)
        # targets = targets.to(device)
        # print(targets.size())

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = 0
        for output,target in zip(outputs, [targets80, targets40, targets20]):  # Combine losses from all detection heads
            loss += criterion(output, target.to(device = device))
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
