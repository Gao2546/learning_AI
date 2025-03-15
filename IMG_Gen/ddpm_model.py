from util.node import *
from util.utils import *
# from util.resize_images import *
from torch.utils.data import DataLoader, Dataset
import torch
import signal
import sys
import random

import torch
import torch.distributed as dist

import torch.multiprocessing as mp
import os

device = torch.device("cuda")
world_size = torch.cuda.device_count()  # Number of GPUs

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
    model_ckp = None#'model/checkpoint/DDPM_T_VQVAE4.pth'
    model_VQVAE = None#"model/checkpoint/VQVAE0.pth"
    signal.signal(signal.SIGINT, signal_handler)
    seed = -1
    set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)
    size = 16*8
    batch_size = 16*32
    path_to_data = './data/104Flower_resized'

    # # train_dataset = YOLODataset_xml(path=path_to_data, class_name=["cat", "dog"], width=size, height=size)
    # transform = transforms.Compose([
    #         # Resize to the desired dimensions
    #         transforms.Resize((size, size)),
    #         # Convert PIL image or numpy array to a tensor
    #         transforms.ToTensor(),
    #         # transforms.Lambda(lambda x:x/255.0),
    #         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
    #             0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    #     ])
    # # train_dataset = datasets.MNIST(
    # #     root='./data', train=True, download=True, transform=transform)
    # train_dataset = datasets.ImageFolder(root=path_to_data,transform=transform)
    # sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler, drop_last=True, num_workers=4)
    
    print("Data loaded")
    model_VQVAE = VQVAETrainer(
        in_c=3, 
        out_c=3, 
        down_sampling_times=2, 
        encode_laten_channel=4, 
        Z_size=16384, 
        load_model_path=model_VQVAE, 
        lr=1e-3
    )
    # model_VQVAE = VQVAETrainer(3, 3, 2, 4, 16384, model_VQVAE, 1e-3)
    # embedding_weights = model_VQVAE.vqvae.embedding.weight.data
    # print(f"Min weight: {embedding_weights.min().item()}")
    # print(f"Max weight: {embedding_weights.max().item()}")
    # model = diffusion_model(
    #     in_c=3, 
    #     out_c=3, 
    #     st_channel=64, 
    #     channel_multi=[1, 2, 4], 
    #     att_channel=64, 
    #     embedding_time_dim=64, 
    #     time_exp=256, 
    #     num_head=1, 
    #     d_model=32, 
    #     num_resbox=2, 
    #     allow_att=[True, True, True], 
    #     concat_up_down=True, 
    #     concat_all_resbox=True, 
    #     down_sampling_times=2, 
    #     encode_laten_channel=4, 
    #     Z_size=16384, 
    #     load_model_path=model_ckp, 
    #     load_model_path_VQVAE=model_VQVAE, 
    #     lr=1e-4
    # )
    # model = diffusion_model(3, 3, 64, [1, 2, 4], 64, 64, 256, 1, 32, 2, [True, True, True], True, True, 2, 4, 16384, model_ckp,model_VQVAE,1e-6 )
    # Print the size of the model
    # model = diffusion_model_No_VQVAE(
    #     in_c=3, 
    #     out_c=3, 
    #     st_channel=64, 
    #     channel_multi=[1, 2, 4], 
    #     att_channel=64, 
    #     embedding_time_dim=64, 
    #     time_exp=256, 
    #     num_head=4, 
    #     d_model=32, 
    #     num_resbox=2, 
    #     allow_att=[True, True, True], 
    #     concat_up_down=True, 
    #     concat_all_resbox=True, 
    #     load_model_path=model_ckp
    # )
    # model = diffusion_model_No_VQVAE(3, 3, 64, [1, 2, 4], 64, 64, 256, 4, 32, 2, [True, True, True], True, True, model_ckp)
    print("Model loaded")
    # model_VQVAE.train(train_loader,100)
    mp.spawn(model_VQVAE.train, args=(world_size, size, path_to_data, batch_size, 100), nprocs=world_size, join=True)
    # model_VQVAE.inference(train_loader,"test")

    # model.train(train_loader=train_loader,num_epoch=100)
    # mp.spawn(model.train, args=(world_size, size, path_to_data, batch_size, 100), nprocs=world_size, join=True)
    # model.inference("test",32*2) #input size of images
    print("Model train finished")
    # train(checkpoint_path=model_ckp, lr=1e-6, batch_size=16*2, num_epochs=100)
    # inference(model_ckp,size=28+4,channel=1)


if __name__ == '__main__':
    main()