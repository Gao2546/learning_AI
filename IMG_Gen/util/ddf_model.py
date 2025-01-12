from node import train , inference
import torch
import signal
import sys
def signal_handler(sig, frame):
    print("Training interrupted by user")
    # Clear all data in GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    sys.exit(0)
def main():
    model_ckp = 'model/checkpoint/DDPM_T01.pth'
    signal.signal(signal.SIGINT, signal_handler)
    train(checkpoint_path=None, lr=2e-5, num_epochs=15)
    # inference(model_ckp)


if __name__ == '__main__':
    main()
