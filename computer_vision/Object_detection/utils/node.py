import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import nms

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
            256), (num_classes + 4) * 3, kernel_size=1, stride=1, padding=0)
        self.detect_p3 = nn.Conv2d(scaled_channels(
            512), (num_classes + 4) * 3, kernel_size=1, stride=1, padding=0)
        self.detect_p5 = nn.Conv2d(scaled_channels(
            512), (num_classes + 4) * 3, kernel_size=1, stride=1, padding=0)

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
        self.mse = nn.MSELoss()
        self.loss_factor_obj = 1.0
        self.loss_factor_cls = 0.5
        self.loss_factor_bbox = 7.5

    def forward(self, preds, targets):
        # Placeholder for loss implementation
        # Objectness score loss
        obj_loss = self.bce(
            preds[..., 4], targets[..., 4])*self.loss_factor_obj
        # Classification loss
        cls_loss = self.bce(
            preds[..., 5:], targets[..., 5:])*self.loss_factor_cls
        # Bounding box loss
        box_loss = self.mse(
            preds[..., :4], targets[..., :4])*self.loss_factor_bbox
        return obj_loss + cls_loss + box_loss


# Non-Maximum Suppression (NMS)
def perform_nms(predictions, conf_thresh=0.5, iou_thresh=0.4):
    """
    :param predictions: Tensor of shape [N, 7] -> [x1, y1, x2, y2, conf, class_score, class_id]
    :param conf_thresh: Confidence threshold for filtering
    :param iou_thresh: IoU threshold for NMS
    """
    boxes = predictions[:, :4]
    scores = predictions[:, 4] * predictions[:, 5]
    keep = nms(boxes, scores, iou_thresh)
    return predictions[keep]


# Optimizer Setup
def setup_optimizer(model):
    return optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)


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
