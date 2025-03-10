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
    model_ckp = 'model/checkpoint/DDPM_T_VQVAE4.pth'
    model_VQVAE = "model/checkpoint/VQVAE0.pth"
    signal.signal(signal.SIGINT, signal_handler)
    seed = -1
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    size = 16*4
    batch_size = 16*16
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
    # model_VQVAE = VQVAETrainer(3, 3, 2, 4, 16384, model_VQVAE, 1e-3)
    # embedding_weights = model_VQVAE.vqvae.embedding.weight.data
    # print(f"Min weight: {embedding_weights.min().item()}")
    # print(f"Max weight: {embedding_weights.max().item()}")
    model = diffusion_model(3, 3, 64, [1, 2, 4], 64, 64, 256, 1, 32, 2, [True, True, True], True, True, 2, 4, 16384, model_ckp,model_VQVAE,1e-6 )
    # model = diffusion_model_No_VQVAE(3, 3, 64, [1, 2, 4], 64, 64, 256, 4, 32, 2, [True, True, True], True, True, model_ckp)
    print("Model loaded")
    # model_VQVAE.train(train_loader,100)
    # model_VQVAE.inference(train_loader,"test")
    # model.train(train_loader=train_loader,num_epoch=100)
    model.inference("test",32*2) #input size of images
    print("Model train finished")
    # train(checkpoint_path=model_ckp, lr=1e-6, batch_size=16*2, num_epochs=100)
    # inference(model_ckp,size=28+4,channel=1)


if __name__ == '__main__':
    main()