import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.node import *
import os
from utils.loaddata import YOLODataset_xml , load_test_images


path_data = "/home/athip/psu/learning_AI/computer_vision/Object_detection/images/test_img"
batch_size = 32//4
num_classes = 2
class_name = ["cat","dog"]
dataset = load_test_images(path=path_data, width=640, height=640)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = YOLOv8(num_classes=num_classes)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

load_path = "/home/athip/psu/learning_AI/computer_vision/Object_detection/models/CatVsDog/CatVsDog25_epoch100"
if load_path != None:
    model, optimizer, scheduler = load_model_and_optimizer(model=model, optimizer=None, lr_schdule=None, filepath=load_path, device=device)

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, real_size) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputsR = [out2bbox(outs) for outs in outputs]
        all_output = torch.cat([outputsR[0].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[1].reshape(batch_size,-1,4 + 1 + num_classes), outputsR[2].reshape(batch_size,-1,4 + 1 + num_classes)], dim=1)
        exp_output = [perform_nms(predictions=outs.view(-1,num_classes+1+4), conf_thresh=0.7, iou_thresh=0.7) for outs in all_output]
        print(exp_output[0])
        # exp_output = [outs[outs[:,4].bool()].view(-1,num_classes+1+4) for outs in all_target]
        # exp_output = [outs[targs[:,4].bool()].view(-1,num_classes+1+4) for outs,targs in zip(all_output,all_target)]
        # print(len(exp_output))
        # print(exp_output[0])
        out_pos = [postprocess(inputs[i],exp_output[i],torch.cat([real_size[0].unsqueeze(1),real_size[1].unsqueeze(1)],dim=1)[i]) for i in range(inputs.size(0))]
        for image, cls_bbox, size in out_pos:
           plot_bbox(image=image, predict=cls_bbox, size=size, class_name=class_name, colors=['green','red'], show=True, linewidth=2)
        # print("/////////////////////////////////////////")
