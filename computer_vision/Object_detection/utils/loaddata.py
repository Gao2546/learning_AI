import os
import torch
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET
from torchvision import transforms
from PIL import Image
import math

class YOLODataset_xml(Dataset):
    def __init__(self, path, class_name, width, height):
        """
        :param images: List of image tensors (C, H, W).
        :param targets: List of target tensors (bounding boxes + classes).
        """
        self.width = width
        self.height = height
        self.target_size = [80, 40, 20]
        self.class_map = {}
        for idx, name in enumerate(class_name):
            self.class_map[name] = idx
        in_path = os.listdir(path=path)
        if len(in_path) == 2:
            data_floder = in_path
        else:
            path = os.path.join(path, "..")
            data_floder = os.listdir(path=path)
        if ".xml" in os.listdir(os.path.join(path, data_floder[0]))[0]:
            self.annotation_floder = os.path.join(path, data_floder[0])
            self.images_floder = os.path.join(path, data_floder[1])
        else:
            self.annotation_floder = os.path.join(path, data_floder[1])
            self.images_floder = os.path.join(path, data_floder[0])

        self.annotation_files = os.listdir(self.annotation_floder)
        self.images_files = os.listdir(self.images_floder)
        self.transform = transforms.Compose([
            # Resize to the desired dimensions
            transforms.Resize((height, width)),
            # Convert PIL image or numpy array to a tensor
            transforms.ToTensor(),
            # transforms.Lambda(lambda x:x/255.0),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
                0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.annotation_files)
    
    def map_size_object(self,Rwidth,Rheight):
        object_ratio = max(Rwidth,Rheight)
        if object_ratio < 0.2:
            return 80
        elif 0.2 <= object_ratio <= 0.35:
            return 40
        else:
            return 20

    def __getitem__(self, idx):
        tree = ET.parse(os.path.join(
            self.annotation_floder, self.annotation_files[idx]))
        # print(tree)
        # root = tree.getroot()
        # print(root)
        # print(root.tag)
        image_file = os.path.join(self.images_floder, tree.find("filename").text)
        width = float(tree.find("size/width").text)
        height = float(tree.find("size/height").text)
        class_idx = self.class_map[tree.find("object/name").text]
        xmin, ymin, xmax, ymax = [float(tree.find("object/bndbox/xmin").text), float(tree.find("object/bndbox/ymin").text),
                                  float(tree.find("object/bndbox/xmax").text), float(tree.find("object/bndbox/ymax").text)]
        # print(f"{xmin} {ymin} {xmax} {ymax}")
        cx, cy, w, h = [(xmin + xmax)/2, (ymin + ymax) /2, xmax - xmin, ymax - ymin]
        # print(f"{cx} {cy} {w} {h}")
        xratio, yratio = (self.width/width,self.height/height)
        invxratio, invyration = (1/xratio, 1/yratio)
        image = Image.open(image_file).convert("RGB")
        image_transform = self.transform(image)
        target = []
        for size in self.target_size:
            T = torch.zeros((size, size, 4 + 1 + len(self.class_map)))
            if self.map_size_object(w/width,h/height) == size:
                x_posi_ratio, y_posi_ratio = (size/width, size/height)
                T[math.ceil(cy*y_posi_ratio), math.ceil(cx*x_posi_ratio), :4] = torch.tensor([cx/width, cy/height, w/width, h/height])
                T[math.ceil(cy*y_posi_ratio), math.ceil(cx*x_posi_ratio), 4] = 1
                T[math.ceil(cy*y_posi_ratio), math.ceil(cx*x_posi_ratio), 5 + class_idx] = 1
            target.append(T)
        return image_transform, target, [width, height]
    
class load_test_images(Dataset):
    def __init__(self, path = None, width = 640, height = 640):
        """
        :param images: List of image tensors (C, H, W).
        :param targets: List of target tensors (bounding boxes + classes).
        """
        self.width = width
        self.height = height
        self.target_size = [80, 40, 20]
        self.images_floder = path

        self.images_files = os.listdir(self.images_floder)
        self.transform = transforms.Compose([
            # Resize to the desired dimensions
            transforms.Resize((height, width)),
            # Convert PIL image or numpy array to a tensor
            transforms.ToTensor(),
            # transforms.Lambda(lambda x:x/255.0),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
                0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.images_floder, self.images_files[idx])
        image = Image.open(image_file).convert("RGB")
        width,height = image.size
        image_transform = self.transform(image)
        return image_transform, [width, height]
