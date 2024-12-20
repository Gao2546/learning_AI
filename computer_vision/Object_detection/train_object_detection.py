import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.node import *
import os
from utils.loaddata import YOLODataset_xml


path_data = "/home/athip/psu/learning_AI/computer_vision/Object_detection/images/CatVsDog"
batch_size = 32//2
num_classes = 2
class_name = ["cat","dog"]
dataset = YOLODataset_xml(path=path_data, class_name=class_name, width=640, height=640)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = YOLOv8(num_classes=num_classes)
criterion = YOLOLoss()
optimizer = setup_optimizer(model=model)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
start_epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Learning rate scheduler
warmup_steps = num_epochs*0.05*4 #5%
max_steps = num_epochs*0.5*4 #50%
scheduler = WarmupCosineScheduler(optimizer, warmup_steps, max_steps, base_lr=0.0005, start_step=start_epoch)

load_path = None#"/home/athip/psu/learning_AI/computer_vision/Object_detection/models/CatVsDog/CatVsDog23_epoch80"
save_path = "/home/athip/psu/learning_AI/computer_vision/Object_detection/models/CatVsDog"
if load_path != None:
    model, optimizer, scheduler = load_model_and_optimizer(model=model, optimizer=optimizer, lr_schdule=scheduler, filepath=load_path, device=device)
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (inputs, (targets80, targets40, targets20), real_size) in enumerate(dataloader):
        inputs = inputs.to(device)
        # targets = targets.to(device)
        # print(targets.size())

        # Forward pass
        outputs = model(inputs)
        outputsR = [out2bbox(outs) for outs in outputs]

        all_output = torch.cat([outputsR[0].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[1].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[2].reshape(batch_size,-1,4 + 1 + num_classes)], dim=1)
        all_target = torch.cat([targets80.reshape(batch_size,-1,4 + 1 + num_classes), targets40.reshape(batch_size,-1,4 + 1 + num_classes), targets20.reshape(batch_size,-1,4 + 1 + num_classes)], dim=1).to(device=device)

        # Compute loss
        # loss = 0
        # for output,target in zip(outputsR, [targets80, targets40, targets20]):  # Combine losses from all detection heads
        #     loss += criterion(output, target.to(device = device))/3

        loss = criterion(all_output,all_target)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update loss
        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx +
              1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    # outputs = None
    outputsR = None
    all_output = None
    all_target = None
    with torch.no_grad():
        if (epoch + 1) % 5 == 0:
            file_name = f"CatVsDog25_epoch{epoch+1}"
            save_model_and_optimizer(model=model, optimizer=optimizer, lr_schdule=scheduler, filepath=os.path.join(save_path, file_name))
            # all_output = torch.cat([outputs[0].reshape(batch_size,-1,4 + 1 + num_classes), outputs[1].reshape(batch_size,-1,4 + 1 + num_classes), outputs[2].reshape(batch_size,-1,4 + 1 + num_classes)], dim=1)
            # exp_output = [perform_nms(predictions=outs.view(-1,num_classes+1+4), conf_thresh=0.5, iou_thresh=0.4) for outs in all_output]
            # out_pos = [postprocess(inputs[i],exp_output[i],torch.cat([real_size[0].unsqueeze(1),real_size[1].unsqueeze(1)],dim=1)[i]) for i in range(inputs.size(0))]
            # for image, cls_bbox, size in out_pos:
            #    plot_bbox(image=image, predict=cls_bbox, size=size, class_name=class_name, colors=['red'], linewidth=2, show=False)
            # outputs = model(inputs)
            # outputs = outputs.clone()
            outputsR = [out2bbox(outs) for outs in outputs]
            all_output = torch.cat([outputsR[0].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[1].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[2].reshape(batch_size,-1,4 + 1 + num_classes)], dim=1)
            all_target = torch.cat([targets80.reshape(batch_size,-1,4 + 1 + num_classes), targets40.reshape(batch_size,-1,4 + 1 + num_classes), targets20.reshape(batch_size,-1,4 + 1 + num_classes)], dim=1)
            print(all_output[all_target[:,:,4].bool()])
            print(all_target[all_target[:,:,4].bool()])
            exp_output = [perform_nms(predictions=outs.view(-1,num_classes+1+4), conf_thresh=0.7, iou_thresh=0.7) for outs in all_output]
            out_pos = [postprocess(inputs[i],exp_output[i],torch.cat([real_size[0].unsqueeze(1),real_size[1].unsqueeze(1)],dim=1)[i]) for i in range(inputs.size(0))]
            for image, cls_bbox, size in out_pos:
               plot_bbox(image=image, predict=cls_bbox, size=size, class_name=class_name, colors=['green','red'], show=False, linewidth=2)
    print(f"Epoch [{
          epoch+1}/{num_epochs}] Completed. Average Loss: {epoch_loss / len(dataloader):.4f}")

print("Training Complete.")
