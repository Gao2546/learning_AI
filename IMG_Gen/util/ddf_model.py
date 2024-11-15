from node import train , inference
def main():
    model_ckp = 'model/checkpoint/DDPM_05.pth'
    train(checkpoint_path='model/checkpoint/DDPM_04.pth', lr=2e-5, num_epochs=15)
    inference(model_ckp)


if __name__ == '__main__':
    main()
    
