import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import nms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import time
from torch.nn.functional import sigmoid , softmax

# Convolutional block with BatchNorm and LeakyReLU


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """
        Initializes the upsampling module using transposed convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Kernel size of the transposed convolution. Default is 4.
            stride (int, optional): Stride of the transposed convolution. Default is 2.
            padding (int, optional): Padding for the transposed convolution. Default is 1.
        """
        super(Upsampling, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        """
        Forward pass of the upsampling module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H_out, W_out).
        """
        return self.conv_transpose(x)


# Bottleneck module (residual block)
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels//2,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(out_channels//2, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


# C2F Bottleneck (formerly CSPBottleneck)
class C2F(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(C2F, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)

        self.blocks = nn.Sequential(
            *[Bottleneck(out_channels // 2, out_channels // 2) for _ in range(num_blocks)]
        )

        self.conv2 = ConvBlock(math.ceil(
            out_channels*0.5*(num_blocks+2)), out_channels, kernel_size=1, stride=1, padding=0)
        # self.conv3 = ConvBlock(out_channels, out_channels,
        #                        kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        list_of_cat = []
        x1 = self.conv1(x)
        # print(torch.split(x1, 2, dim=1))
        x_split = torch.split(x1, [x1.size(1)//2, x1.size(1)//2], dim=1)
        x1_1 = x_split[0]
        x1_2 = x_split[1]
        list_of_cat.append(x1_1)
        list_of_cat.append(x1_2)
        for block in self.blocks:
            x1_2 = block(x1_2)
            list_of_cat.append(x1_2)
        x2 = torch.cat(list_of_cat, dim=1)
        return self.conv2(x2)


# SPPF Layer
class SPPFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(5, 5, 5)):
        super(SPPFLayer, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.pools = nn.ModuleList([nn.MaxPool2d(
            kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes])
        self.conv2 = ConvBlock(
            out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        list_of_cat = []
        x = self.conv1(x)
        list_of_cat.append(x)
        for pool in self.pools:
            x = pool(x)
            list_of_cat.append(x)
        x = torch.cat(list_of_cat, dim=1)
        return self.conv2(x)


# YOLOv8 Backbone
class YOLOv8Backbone(nn.Module):
    def __init__(self, d=1.0, w=1.0, r=1.0):
        """
        YOLOv8 Backbone with depth and width multipliers.
        :param d: Depth multiplier to scale the number of blocks.
        :param w: Width multiplier to scale the number of channels.
        """
        super(YOLOv8Backbone, self).__init__()

        def scaled_channels(channels, plus=0):
            return max(1, int(channels * (w+plus)))

        def scaled_depth(depth, plus=0):
            return max(1, math.ceil(depth * (d+plus)))

        def scaled_ratio(ratio, plus=0):
            return max(1, math.ceil(ratio * (r+plus)))

        self.layer1 = ConvBlock(3, scaled_channels(
            64), kernel_size=3, stride=2, padding=1)

        self.layer_c2 = ConvBlock(scaled_channels(64), scaled_channels(
            128), kernel_size=3, stride=2, padding=1)
        self.layer2 = C2F(scaled_channels(128), scaled_channels(
            128), num_blocks=scaled_depth(3))

        self.layer_c3 = ConvBlock(scaled_channels(128), scaled_channels(
            256), kernel_size=3, stride=2, padding=1)
        self.layer3 = C2F(scaled_channels(256), scaled_channels(
            256), num_blocks=scaled_depth(6))

        self.layer_c4 = ConvBlock(scaled_channels(256), scaled_channels(
            512), kernel_size=3, stride=2, padding=1)
        self.layer4 = C2F(scaled_channels(512), scaled_channels(
            512), num_blocks=scaled_depth(6))

        self.layer_c5 = ConvBlock(scaled_channels(512), scaled_ratio(
            scaled_channels(512)), kernel_size=3, stride=2, padding=1)
        self.layer5 = C2F(scaled_ratio(scaled_channels(512)), scaled_ratio(
            scaled_channels(512)), num_blocks=scaled_depth(3))

        self.sppf = SPPFLayer(scaled_ratio(scaled_channels(
            512)), scaled_ratio(scaled_channels(512)))

    def forward(self, x):
        x1 = self.layer2(self.layer_c2(self.layer1(x)))  # Low-level features
        x2 = self.layer3(self.layer_c3(x1))             # Mid-level features
        x3 = self.layer4(self.layer_c4(x2))             # High-level features
        x4 = self.layer5(self.layer_c5(x3))             # Deeper features
        # SPPF-enhanced features
        x5 = self.sppf(x4)
        return x2, x3, x5


# YOLOv8 Detection Head
class YOLOv8HeadFull(nn.Module):
    def __init__(self, num_classes, d=1.0, w=1.0, r=1.0):
        super(YOLOv8HeadFull, self).__init__()

        def scaled_channels(channels, plus=0):
            return max(1, int(channels * (w+plus)))

        def scaled_depth(depth, plus=0):
            return max(1, math.ceil(depth * (d+plus)))

        def scaled_ratio(ratio, plus=0):
            return max(1, math.ceil(ratio * (r+plus)))

        # P5 (Deeper features)
        self.conv1_p5 = ConvBlock(scaled_channels(512), scaled_channels(
            512), kernel_size=3, stride=2, padding=1)
        self.concat_p5 = ConvBlock(scaled_ratio(scaled_channels(512), plus=1), scaled_ratio(
            scaled_channels(512), plus=1), kernel_size=1, stride=1, padding=0)
        self.c2f_p5 = C2F(scaled_ratio(scaled_channels(512), plus=1),
                          scaled_channels(512), num_blocks=scaled_depth(3))

        # P3 (Intermediate features)
        self.conv1_p3 = ConvBlock(scaled_channels(256), scaled_channels(
            256), kernel_size=3, stride=2, padding=1)
        self.concat_p3 = ConvBlock(scaled_channels(
            512 + 256), scaled_channels(512), kernel_size=1, stride=1, padding=0)
        self.c2f_p3 = C2F(scaled_channels(512), scaled_channels(
            512), num_blocks=scaled_depth(3))

        # P2 (Shallow features)
        self.upsample_p2_1 = Upsampling(scaled_ratio(
            scaled_channels(512)), scaled_ratio(scaled_channels(512)))
        self.concat_p2_1 = ConvBlock(scaled_ratio(scaled_channels(512), plus=1), scaled_ratio(
            scaled_channels(512), plus=1), kernel_size=1, stride=1, padding=0)
        self.c2f_p2_1 = C2F(scaled_ratio(scaled_channels(
            512), plus=1), scaled_channels(512), num_blocks=scaled_depth(3))

        self.upsample_p2_2 = Upsampling(
            scaled_channels(512), scaled_channels(256))
        self.concat_p2_2 = ConvBlock(scaled_channels(
            256 + 256), scaled_channels(512), kernel_size=1, stride=1, padding=0)
        self.c2f_p2_2 = C2F(scaled_channels(512), scaled_channels(
            256), num_blocks=scaled_depth(3))
        # self.conv1_p3 = ConvBlock(scaled_channels(128), scaled_channels(
        #     128), kernel_size=1, stride=1, padding=0)
        # self.c2f_p3 = C2F(scaled_channels(32), scaled_channels(
        #     32), num_blocks=scaled_depth(3))

        # Final Detect layers
        self.detect_p2 = nn.Conv2d(scaled_channels(
            # (*1 is free anchor box)
            256), (num_classes + 4 + 1) * 1, kernel_size=1, stride=1, padding=0)
        self.detect_p3 = nn.Conv2d(scaled_channels(
            # (*1 is free anchor box)
            512), (num_classes + 4 + 1) * 1, kernel_size=1, stride=1, padding=0)
        self.detect_p5 = nn.Conv2d(scaled_channels(
            # (*1 is free anchor box)
            512), (num_classes + 4 + 1) * 1, kernel_size=1, stride=1, padding=0)

    def forward(self, p5, p3, p2):

        # P2
        p2_1 = self.upsample_p2_1(p5)
        p2_2 = torch.cat([p3, p2_1], dim=1)
        p2_3 = self.concat_p2_1(p2_2)
        # print(p2_1.size())
        # print(p2_2.size())
        # print(p2_3.size())
        p2_4 = self.c2f_p2_1(p2_3)  # -->concat p3

        P2_5 = self.upsample_p2_2(p2_4)
        P2_6 = torch.cat([p2, P2_5], dim=1)
        p2_7 = self.concat_p2_2(P2_6)
        p2_8 = self.c2f_p2_2(p2_7)  # -->detect p2 , conv1_p3

        # p3
        p3_1 = self.conv1_p3(p2_8)
        p3_2 = torch.cat([p2_4, p3_1], dim=1)
        p3_3 = self.concat_p3(p3_2)
        p3_4 = self.c2f_p3(p3_3)  # -->detect p3 . conv1_p5

        # p5
        p5_1 = self.conv1_p5(p3_4)
        p5_2 = torch.cat([p5, p5_1], dim=1)
        p5_3 = self.concat_p5(p5_2)
        p5_4 = self.c2f_p5(p5_3)  # -->detect p5

        # Detect layers
        detect_p2 = self.detect_p2(p2_8).permute(0, 2, 3, 1)  # Small objects
        detect_p3 = self.detect_p3(p3_4).permute(0, 2, 3, 1)  # Medium objects
        detect_p5 = self.detect_p5(p5_4).permute(0, 2, 3, 1)  # Large objects

        return detect_p2, detect_p3, detect_p5


class YOLOv8(nn.Module):
    def __init__(self, num_classes=80, d=0.33, w=0.25, r=2):
        """
        Full YOLOv8 model with depth and width multipliers.
        :param num_classes: Number of object classes.
        :param d: Depth multiplier.
        :param w: Width multiplier.
        """
        super(YOLOv8, self).__init__()
        self.backbone = YOLOv8Backbone(d=d, w=w, r=r)
        self.head = YOLOv8HeadFull(num_classes=num_classes, d=d, w=w, r=r)

    def forward(self, x):
        p2, p3, p5 = self.backbone(x)
        return self.head(p5, p3, p2)


# Loss Function (Simplified)
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        self.mse = nn.MSELoss()
        self.loss_factor_obj = 2.5 #2.5**
        self.loss_factor_cls = 1.5
        self.loss_factor_bbox = 7.5

    def forward(self, preds, targets):
        # Placeholder for loss implementation
        # Objectness score loss
        obj_loss = self.focal_loss(
            preds[..., 4], targets[..., 4])*self.loss_factor_obj
        # Classification loss
        if torch.sum(targets) != 0:
            cls_loss = self.bce(
                preds[..., 5:][targets[..., 4].bool()], targets[..., 5:][targets[..., 4].bool()])*self.loss_factor_cls
            # Bounding box loss
            box_loss = self.mse(
                preds[..., :4][targets[..., 4].bool()], targets[..., :4][targets[..., 4].bool()])*self.loss_factor_bbox
        else:
            cls_loss = 0
            box_loss = 0
        print(obj_loss)
        print(cls_loss)
        print(box_loss)
        print("--------------------------------")
        return obj_loss + cls_loss + box_loss


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=0.0025):  # 3/((80*80) + (40*40) + (20*20))*10
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, inputs, targets):
#         BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-BCE_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return focal_loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1-(0.005)): #0.000025* 0.00025 0.0005** 0.001 0.005** 0.01 0.05 0.25
        """
        Focal Loss for classification tasks with imbalanced datasets.

        Args:
            gamma (float): Focusing parameter to reduce the loss for well-classified examples.
            alpha (float): Scaling factor for balancing positive and negative examples.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits from the model, expected shape (N, *).
            targets (torch.Tensor): Ground truth labels, same shape as inputs.

        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        # self.alpha = 1 - (torch.sum(targets) / (targets.size(1) * targets.size(1)))
        # BCE with logits computes logits internally; reduction is set to 'none' to compute manually.
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        # Compute the probability of the predictions
        probs = torch.sigmoid(inputs)

        # Define pt for positive and negative cases
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Apply Focal Loss formula
        focal_loss = ((targets*self.alpha) + ((1 - targets)*(1-self.alpha))) * (((1 - pt) ** self.gamma) * bce_loss)

        # Return the mean loss
        return focal_loss.sum() #/ (targets.sum() if targets.sum() != 0 else 1)


# Non-Maximum Suppression (NMS)
def perform_nms(predictions, conf_thresh=0.7, iou_thresh=0.7):
    """
    :param predictions: Tensor of shape [N, 7] -> [x1, y1, x2, y2, conf, class_score, class_id]
    :param conf_thresh: Confidence threshold for filtering
    :param iou_thresh: IoU threshold for NMS
    """
    boxes = torch.tensor(
        list(map(cxcywh2xy1xy2, predictions[:, :4].detach().tolist())), device=0)
    # print(torch.tensor(softmax(predictions[:, 4].clone(), dim = 0).tolist()))
    predictions = predictions.clone()
    predictions[:, 4] = torch.clamp(predictions[:, 4].clone(),0,100)/(predictions[:, 4].max())
    scores = predictions[:, 4]
    # print(predictions[:, 4].max())
    # scores = torch.where(
        # predictions[:, 4] > conf_thresh, predictions[:, 4], 0.0)  # * \
    # torch.max(predictions[:, 5:], dim=1).values
    # scores = predictions[:, 4]
    keep = nms(boxes, scores, iou_thresh)[:]
    predictions = predictions[keep]
    return predictions[predictions[:, 4] > conf_thresh]


# Optimizer Setup
def setup_optimizer(model):
    return optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)


def save_model_and_optimizer(model, optimizer, lr_schdule, filepath):
    """
    Saves the state dictionaries of a model and its optimizer to a file.

    Parameters:
    model (torch.nn.Module): The PyTorch model.
    optimizer (torch.optim.Optimizer): The optimizer for the model.
    filepath (str): The file path to save the state dictionaries.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_schdule_step': lr_schdule.current_step
    }
    torch.save(checkpoint, filepath)
    print(f"Model and optimizer state dictionaries saved to {filepath}")


def load_model_and_optimizer(model, optimizer, lr_schdule, filepath, device='cpu'):
    """
    Loads the state dictionaries of a model and its optimizer from a file.

    Parameters:
    model (torch.nn.Module): The PyTorch model instance.
    optimizer (torch.optim.Optimizer): The optimizer for the model.
    filepath (str): The file path to load the state dictionaries from.
    device (str): The device to map the state dictionaries to ('cpu' or 'cuda').

    Returns:
    tuple: The model and optimizer with loaded states.
    """
    checkpoint = torch.load(filepath, map_location=device)
    if optimizer == None or lr_schdule == None:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_schdule.current_step = checkpoint['lr_schdule_step']
    print(f"Model and optimizer state dictionaries loaded from {filepath}")
    return model, optimizer, lr_schdule


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, start_step):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=(max_steps - warmup_steps))
        self.current_step = 0
        if start_step != None:
            self.current_step = start_step
            self.cosine_scheduler.step(start_step)

    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing
            self.cosine_scheduler.step()

        self.current_step += 1


def cxcywh2xy1xy2(bbox):
    cx, cy, width, height = bbox
    x_min, y_min, x_max, y_max = [cx - width/2,
                                  cy - height/2, cx + width/2, cy + height/2]
    return x_min, y_min, x_max, y_max


def cxcywh2xywh(bbox):
    cx, cy, width, height = bbox
    x_min, y_min = [cx - (width/2), cy - (height/2)]
    return x_min, y_min, width, height


def plot_bbox(image, predict, size, class_name=None, colors=None, linewidth=2, show=False):
    predict = predict.cpu().detach()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    bboxes = predict[:, :4]
    # Plot each bounding box
    for i, bbox in enumerate(bboxes):
        # print(bbox)
        x_min, y_min, width, height = cxcywh2xywh(bbox=bbox)
        # x_min, y_min = [cx - width/2, cy - height/2]
        x_min = x_min*size[0]
        y_min = y_min*size[1]
        # cx = cx*size[0]
        # cy = cy*size[1]
        width = width*size[0]
        height = height*size[1]
        # Default to red if no color specified
        color = colors[i] if colors and i < len(colors) else 'red'
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        class_idx = torch.argmax(predict[i, 5:], dim=-1)
        # print(x_min)
        # print(y_min)
        # print(width)
        # print(height)
        # print("-----------------")
        # Add label if provided
        if class_name and class_idx < len(class_name):
            ax.text(
                x_min, y_min - 5, class_name[class_idx], color=color, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

    plt.axis('off')
    if show:
        plt.show()
    else:
        plt.savefig(
            f"/home/athip/psu/learning_AI/computer_vision/Object_detection/output/CatVsDog/CatVsDog{time.time()}.png")


def postprocess(image, out_pre, size):
    transform = transforms.Compose([
        # Resize to the desired dimensions
        transforms.Resize((int(size[1]), int(size[0]))),
        # Convert PIL image or numpy array to a tensor
        transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                             std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.ToPILImage(),
    ])
    image = transform(image)
    # temp_x = out_pre[:, :4:2] * size[0]
    # temp_y = out_pre[:, 1:4:2] * size[1]
    # out_pre[:, :4:2] = temp_x
    # out_pre[:, 1:4:2] = temp_y
    return [image, out_pre, size]


def out2bbox(predict):
    size = predict.size(2)
    batch = predict.size(0)
    tlcorner = torch.tensor([[[[i] for i in range(size)]
                            for j in range(size)] for k in range(batch)], device=0)
    x_tlcorner = tlcorner
    y_tlcorner = tlcorner.permute(0, 2, 1, 3)
    predict[:, :, :, 0:1] = (
        sigmoid(predict[:, :, :, 0:1].clone()) + x_tlcorner)/size
    predict[:, :, :, 1:2] = (
        sigmoid(predict[:, :, :, 1:2].clone()) + y_tlcorner)/size
    predict[:, :, :, 2:3] = torch.exp(predict[:, :, :, 2:3].clone())
    predict[:, :, :, 3:4] = torch.exp(predict[:, :, :, 3:4].clone())
    return predict



# Example Usage
# if __name__ == "__main__":
#     device = 0
#     model = YOLOv8(num_classes=80).to(device=device)
#     optimizer = setup_optimizer(model)
#     criterion = YOLOLoss()

#     input_tensor = torch.randn(1, 3, 416, 416).to(device=device)
#     # Example target tensor for 3 heads
#     targets = torch.randn(1, 3, 13, 13, 85).to(device=device)

#     outputs = model(input_tensor)

#     total_loss = 0
#     for output in outputs:
#         total_loss += criterion(output, targets)

#     total_loss.backward()
#     optimizer.step()

#     print("Total Loss:", total_loss.item())
