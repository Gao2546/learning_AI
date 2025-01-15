from util.node import *
from util.utils import *
# from util.resize_images import *
from torch.utils.data import DataLoader, Dataset
import torch
import signal
import sys
import random

def set_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

def signal_handler(sig, frame):
    print("Training interrupted by user")
    # Clear all data in GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sys.exit(0)


def main():
    model_ckp = None#'model/checkpoint/DDPM_T01.pth'
    signal.signal(signal.SIGINT, signal_handler)
    seed = -1
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    size = 16*8
    batch_size = 1
    path_to_data = './data/104Flower_resized'

    # train_dataset = YOLODataset_xml(path=path_to_data, class_name=["cat", "dog"], width=size, height=size)
    transform = transforms.Compose([
            # Resize to the desired dimensions
            transforms.Resize((size, size)),
            # Convert PIL image or numpy array to a tensor
            transforms.ToTensor(),
            # transforms.Lambda(lambda x:x/255.0),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
                0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
    # train_dataset = datasets.MNIST(
    #     root='./data', train=True, download=True, transform=transform)
    train_dataset = datasets.ImageFolder(root=path_to_data,transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    print("Data loaded")
    # model = diffusion_model(3, 3, 256, [1, 2, 4], 64, 64, 256, 1, 32, 2, [True, True, True], True, True, 2, 4, 16384, model_ckp )
    model = diffusion_model_No_VQVAE(3, 3, 64, [1, 2, 4], 64, 64, 256, 4, 32, 2, [True, True, True], True, True, model_ckp)
    print("Model loaded")
    model.train(train_loader=train_loader,num_epoch=100)
    print("Model train finished")
    # train(checkpoint_path=model_ckp, lr=1e-6, batch_size=16*2, num_epochs=100)
    # inference(model_ckp,size=28+4,channel=1)


if __name__ == '__main__':
    main()