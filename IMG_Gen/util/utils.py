import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets,transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from IPython.display import clear_output
from torch import optim
import math
import tqdm
import numpy as np
import gc
from torch.cuda import amp
from mpl_toolkits.axes_grid1 import ImageGrid
device = torch.device("cuda")

import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    scale = 1000 / timesteps
    start = scale*start
    end   = scale*end
    return torch.linspace(start, end, timesteps)

# def cosine_beta_schedule(timestep, initial_lr = 0.0001, final_lr = 0.02):
#     return final_lr + 0.5 * (initial_lr - final_lr) * (1 + torch.cos(math.pi * torch.linspace(0, timestep,timestep)))

def cosine_beta_schedule(num_timesteps, s=0.008):
  def f(t):
    return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
  x = torch.linspace(0, num_timesteps, num_timesteps + 1)
  alphas_cumprod = f(x) / f(torch.tensor([0]))
  betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  betas = torch.clip(betas, 0.0001, 0.999)
  return betas

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 1000
betas = linear_beta_schedule(timesteps=T)
# betas = cosine_beta_schedule(num_timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = 1.0/torch.sqrt(alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

import torch
import torch.nn.functional as F

def reconstruction_loss(input_data, reconstructed_data, reduction='mean'):

    if reduction == 'none':
        return F.mse_loss(input_data, reconstructed_data, reduction='none')
    else:
        return F.mse_loss(input_data, reconstructed_data, reduction=reduction)

def kl_divergence_loss(mu, logvar):

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def lvlb_loss(input_data, reconstructed_data, mu, logvar, beta=1.0, reduction='mean'):

    recon_loss = reconstruction_loss(input_data, reconstructed_data, reduction=reduction)
    kl_loss = kl_divergence_loss(mu, logvar)

    if reduction == 'none':
        return recon_loss + beta * kl_loss
    else:
        return recon_loss.mean() + beta * kl_loss.mean()

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy,t)
    # noise_pred = model(x_noisy, t)
    # return F.mse_loss(noise_pred,noise)
    return reconstruction_loss(input_data=noise,reconstructed_data=noise_pred)

def show_tensor_image_t(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        return reverse_transforms(image[0, :, :, :])
    return reverse_transforms(image)
    # plt.imshow(reverse_transforms(image))
@torch.no_grad()
def sample_timestep(x, t, time_step, Denoise_models):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    with torch.no_grad():
        pre_noises = Denoise_models(x,t)
    # noises = Denoise_models(x, t)
    model_mean = sqrt_recip_alphas_t * (
        x - (betas_t / sqrt_one_minus_alphas_cumprod_t * pre_noises)
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    # if t == 0:
    #     # As pointed out by Luis Pereira (see YouTube comment)
    #     # The t's are offset from the t's in the paper
    #     return model_mean
    # else:
    #     noise = torch.randn_like(x)
    #     return model_mean + torch.sqrt(posterior_variance_t) * noise
    noise = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)
    return model_mean + torch.sqrt(betas_t) * noise
    # return model_mean

@torch.no_grad()
def sample_plot_image(Encode_Decode,Denoise_model,names,size):
    # Sample noise
    img_size = size//2#IMG_SIZE
    img = torch.randn((5, 4, img_size, img_size), device=device)
    # plt.figure(figsize=(15,15))
    # plt.axis('off')
    # num_images = 10
    # stepsize = int(T/num_images)

    for time_step in range(1,1000)[::-1]:
        t = torch.ones(5,device=device, dtype=torch.long) * time_step
        img = sample_timestep(img, t , time_step, Denoise_model)
        # Edit: This is to maintain the natural range of the distribution
    img = torch.clamp(img, -1.0, 1.0)
    with torch.no_grad():
        img = Encode_Decode.post_quant_conv(img)
        for dec in Encode_Decode.decode:
            img = dec(img)
        img = Encode_Decode.output_layer(img)
        # img = decode(img)
    fig = plt.figure(1,clear=True)
    grid = ImageGrid(fig,rect=111 ,  # similar to subplot(111)
                     nrows_ncols=(1, 5),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.

                     )
    for ax, im in zip(grid, img.to("cpu")):
        # Iterating over the grid returns the Axes.
        ax.imshow(show_tensor_image(im))
    # for im in range(img.size(0)):
    #     plt.subplot(4, 8, im+1)
    #     show_tensor_image(img[im].detach().cpu())
    # plt.show()
    plt.savefig(f"output/sample_{names}.png")

@torch.no_grad()
def sample_plot_image_no_VQVAE(Denoise_model, names, size):
    # Sample noise
    img_size = size
    img = torch.randn((5, 3, img_size, img_size), device=device)
    # num_images = 10
    # stepsize = int(1000 / num_images)

    for time_step in range(1, 1000)[::-1]:
        t = torch.ones(5, device=device, dtype=torch.long) * time_step
        img = sample_timestep(img, t, time_step, Denoise_model)
    img = torch.clamp(img, -1.0, 1.0)
    fig = plt.figure(1, clear=True)
    grid = ImageGrid(fig, rect=111, nrows_ncols=(1, 5), axes_pad=0.1)

    for ax, im in zip(grid, img.to("cpu")):
        ax.imshow(show_tensor_image(im))
    plt.savefig(f"output/sample_no_VQVAE_{names}.png")

def show_img_VAE(batch,recon,names):
    fig = plt.figure(1,clear=True)
    grid = ImageGrid(fig,rect=111 ,  # similar to subplot(111)
                     nrows_ncols=(4,8 ),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.

                     )
    for ax, im in zip(grid, batch.to("cpu")):
        # Iterating over the grid returns the Axes.
        ax.imshow(show_tensor_image(im))

    fig = plt.figure(2,clear=True)
    grid = ImageGrid(fig,rect=111 ,  # similar to subplot(111)
                     nrows_ncols=(4,8 ),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.

                     )
    for ax, im in zip(grid, recon.to("cpu")):
        # Iterating over the grid returns the Axes.
        ax.imshow(show_tensor_image(im))
    plt.savefig(f"output/VQVAE{names}.png")
    # for im in range(img.size(0)):
    #     plt.subplot(4, 8, im+1)
    #     show_tensor_image(img[im].detach().cpu())
    # plt.show()
