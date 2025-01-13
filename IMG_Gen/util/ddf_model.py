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
    train(checkpoint_path=model_ckp, lr=2e-5, batch_size=16, num_epochs=100)
    # inference(model_ckp,size=28+4,channel=1)


if __name__ == '__main__':
    main()
