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
from util.utils import *
import random
device = torch.device("cuda")

class Attention(nn.Module):
    def __init__(self,num_head, channels):
        super().__init__()
        self.channels = channels
        self.num_head = num_head

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=num_head, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.channels, H * W).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.swapaxes(2, 1).view(B, self.channels, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h

class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,mlp_dim,num_head,d_model,allow_att=False) -> None:
        super(ResBlock,self).__init__()
        self.time_mlp = nn.Linear(mlp_dim,out_c)
        self.conv2d1 = nn.Conv2d(in_c,out_c,3,1,padding="same")
        if (in_c % 8) != 0:
            self.groupnorm1 = nn.GroupNorm(num_groups=1,num_channels=in_c)
        else:
            self.groupnorm1 = nn.GroupNorm(num_groups=8,num_channels=in_c)
        self.conv2d2 = nn.Conv2d(out_c,out_c,3,1,padding="same")
        self.groupnorm2 = nn.GroupNorm(num_groups=8,num_channels=out_c)
        # self.conv2d3 = nn.Conv2d(out_c,out_c,3,1,1)
        if in_c != out_c:
            self.match_input = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()
        self.silu = nn.SiLU()
        self.drop = nn.Dropout2d(0.1)
        if allow_att:
            self.attention = Attention(num_head=num_head,channels=out_c)
        else:
            self.attention = nn.Identity()
    def forward(self,x,t):
        x_z = self.silu(self.groupnorm1(x))
        x_z = self.conv2d1(x_z)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(..., ) + (None, ) * 2]
        x_z = x_z + time_emb
        x_z = self.silu(self.groupnorm2(x_z))
        x_z = self.drop(x_z)
        x_z = self.conv2d2(x_z)
        x_z = x_z + self.match_input(x)
        x_z = self.attention(x_z)
        return x_z
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args):
        return self.upsample(x)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)
class UNet(nn.Module):
    def __init__(self,in_c,out_c,st_channel,channel_multi,att_channel,embedding_time_dim,time_exp,num_head,d_model,num_resbox,allow_att,concat_up_down,concat_all_resbox) -> None:
        super(UNet,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.st_channel = st_channel
        self.channel_multi = channel_multi
        self.att_channel = att_channel
        self.embedding_time_dim = embedding_time_dim
        self.num_resbox = num_resbox
        self.allow_att = allow_att
        self.concact_up_down = concat_up_down
        self.concat_all_resbox = concat_all_resbox

        self.time_mlp = SinusoidalPositionEmbeddings(time_emb_dims=embedding_time_dim, time_emb_dims_exp=time_exp)


        # self.frist = nn.Sequential(nn.ConvTranspose2d(in_c,in_c//2,4,2,1),nn.BatchNorm2d(in_c//2),nn.SiLU(),
        #                            nn.ConvTranspose2d(in_c//2,st_channel,4,2,1),nn.BatchNorm2d(st_channel),nn.SiLU(),
        #                            nn.Conv2d(in_channels=st_channel, out_channels=st_channel, kernel_size=4, stride=4, padding=0))

        # self.first = nn.Conv2d(in_c,st_channel,3,1,1)

        self.first = nn.Conv2d(in_c,st_channel,3,1,padding="same")

        self.down_layer = nn.ModuleList()
        if concat_up_down:
            crr_channel = [st_channel]
            res_up = 1
        else:
            res_up = 0
            crr_channel = []
        input_channel = self.st_channel
        for i in range(len(self.channel_multi)):
            output_channel = self.st_channel*self.channel_multi[i]
            for _ in range(self.num_resbox):
                self.down_layer.append(ResBlock(in_c=input_channel,out_c=output_channel,mlp_dim=time_exp,num_head=num_head,d_model=d_model,allow_att=allow_att[i]))
                input_channel = output_channel
                if concat_all_resbox:
                    crr_channel.append(input_channel)
            if not concat_all_resbox:
                crr_channel.append(input_channel)

            if i != (len(self.channel_multi)-1):
                self.down_layer.append(DownSample(channels=input_channel))
                if concat_up_down:
                    crr_channel.append(input_channel)

        self.mid_layer = nn.ModuleList([ResBlock(in_c=input_channel,out_c=input_channel,mlp_dim=time_exp,num_head=num_head,d_model=d_model,allow_att=True),
                                       ResBlock(in_c=input_channel,out_c=input_channel,mlp_dim=time_exp,num_head=num_head,d_model=d_model,allow_att=False)])

        self.up_layer = nn.ModuleList()
        for i in reversed(range(len(self.channel_multi))):
            output_channel = self.st_channel*self.channel_multi[i]
            if (not concat_all_resbox) and (not concat_up_down):
                concat_channel = crr_channel.pop()
            for _ in range(self.num_resbox+res_up):
                if (concat_all_resbox) or concat_up_down:
                    concat_channel = crr_channel.pop()
                self.up_layer.append(ResBlock(in_c=input_channel+concat_channel,out_c=output_channel,mlp_dim=time_exp,num_head=num_head,d_model=d_model,allow_att=allow_att[i]))
                input_channel = output_channel
                concat_channel = 0

            if i != 0:
                self.up_layer.append(UpSample(channels=input_channel))

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=input_channel),
            nn.SiLU(),
            nn.Conv2d(in_channels=input_channel, out_channels=out_c, kernel_size=3, stride=1, padding="same"),
        )
        # self.tanh = nn.Tanh()

    def forward(self,x,t):
        t = self.time_mlp(t)
        h = self.first(x)
        b = 0
        if self.concact_up_down:
            reserve = [h]
        else:
            reserve = []
        for down in self.down_layer:
            h = down(h,t)
            if isinstance(down,ResBlock):
                b += 1
            if (isinstance(down,ResBlock) and (b == self.num_resbox)) or (isinstance(down,ResBlock) and self.concat_all_resbox) or (self.concact_up_down) :
                b = 0
                reserve.append(h)
        for mid in self.mid_layer:
            h = mid(h,t)
        b = 1
        for up in self.up_layer:
            if isinstance(up,ResBlock):
                b += 1
            if (isinstance(up,ResBlock) and (b == self.num_resbox)) or (self.concact_up_down and isinstance(up,ResBlock)):
                h_0 = reserve.pop()
                h = torch.concat([h,h_0],dim=1)
                b = 0
            h = up(h,t)
        out = self.out(h)
        return out
    
class VQVAE(nn.Module):
    def __init__(self,in_c,out_c,st_c,down_sampling_times,encode_laten_channel,Z_size) -> None:
        super(VQVAE,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.down_sampling_times = down_sampling_times
        self.st_c = st_c

        self.beta = 0.2

        self.input_layer = nn.Sequential(nn.Conv2d(in_c,st_c,1,1,0),nn.BatchNorm2d(st_c),nn.SiLU())

        self.encode = nn.ModuleList()
        input_channel = st_c
        for layer in range(down_sampling_times):
            output_channel = input_channel*2
            self.encode.append(nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=4,stride=2,padding=1))
            self.encode.append(nn.BatchNorm2d(num_features=output_channel))
            self.encode.append(nn.SiLU())
            # self.encode.append(nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=1,stride=1,padding=0))
            # self.encode.append(nn.BatchNorm2d(num_features=output_channel))
            # self.encode.append(nn.SiLU())
            input_channel = output_channel

        # for layer in range(down_sampling_times):
        #     output_channel = input_channel//2
        #     self.encode.append(nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,stride=1,padding=1))
        #     self.encode.append(nn.BatchNorm2d(num_features=output_channel))
        #     self.encode.append(nn.SiLU())
        #     input_channel = output_channel

        self.pre_quant_conv = nn.Conv2d(input_channel, encode_laten_channel, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=Z_size, embedding_dim=encode_laten_channel)
        self.embedding.weight.data.uniform_(-1, 1)
        self.embedding.weight.requires_grad = False
        self.post_quant_conv = nn.Conv2d(encode_laten_channel, input_channel, kernel_size=1)

        self.decode = nn.ModuleList()
        input_channel = output_channel
        for layer in range(down_sampling_times):
            output_channel = input_channel//2
            self.decode.append(nn.ConvTranspose2d(in_channels=input_channel,out_channels=output_channel,kernel_size=4,stride=2,padding=1))
            self.decode.append(nn.BatchNorm2d(num_features=output_channel))
            self.decode.append(nn.SiLU())
            # self.decode.append(nn.Conv2d(in_channels=output_channel,out_channels=output_channel,kernel_size=1,stride=1,padding=0))
            # self.decode.append(nn.BatchNorm2d(num_features=output_channel))
            # self.decode.append(nn.SiLU())
            input_channel = output_channel

        self.output_layer = nn.Sequential(#nn.Conv2d(output_channel,output_channel,3,1,padding="same"),
                                        #   nn.BatchNorm2d(num_features=output_channel),
                                        #   nn.SiLU(),
                                        #   nn.Conv2d(output_channel,output_channel,3,1,padding="same"),
                                        #   nn.BatchNorm2d(num_features=output_channel),
                                        #   nn.SiLU(),
                                        #   nn.Conv2d(output_channel,output_channel,3,1,padding="same"),
                                        #   nn.BatchNorm2d(num_features=output_channel),
                                        #   nn.SiLU(),
                                        #   nn.Conv2d(output_channel,output_channel,3,1,padding="same"),
                                        #   nn.BatchNorm2d(num_features=output_channel),
                                        #   nn.SiLU(),
                                          nn.Conv2d(output_channel,out_c,1,1,0),nn.Tanh())

        # self.encode = Encode(in_c,st_c,down_sampling_times,encode_laten_channel)
        # output_channel = st_c*(2**down_sampling_times)
        # self.decode = Decode(out_c,output_channel,down_sampling_times,encode_laten_channel)
    def forward(self,x):
        x = self.input_layer(x)
        for encode in self.encode:
            x = encode(x)
        quant_input = self.pre_quant_conv(x)

        ## Quantization
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        # print(quant_input.shape)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
        # print(quant_input.shape)

        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        # print(self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)).shape)
        # print(dist.shape)

        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)
        # print(min_encoding_indices.shape)

        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        # print(quant_out.min(),quant_out.max())
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        # print(quant_input.shape)
        # print(quant_out.shape)

        # Compute losses
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        # print(commitment_loss)
        # print(codebook_loss)
        quantize_losses = codebook_loss + self.beta*commitment_loss

        # Ensure straight through gradient
        quant_out = quant_input + (quant_out - quant_input).detach()
        # print(quant_out.min(),quant_out.max())

        # Reshaping back to original input shape
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        # print(quant_out.min(),quant_out.max())

        decoder_input = self.post_quant_conv(quant_out)
        for decode in self.decode:
            decoder_input = decode(decoder_input)
        output = self.output_layer(decoder_input)

        # x = self.encode(x)
        # x = self.decode(x)
        return output,quantize_losses

def codebook(quant_input,embedding):
    B, C, H, W = quant_input.shape
    quant_input = quant_input.permute(0, 2, 3, 1)
    # print(quant_input.shape)
    quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))
    # print(quant_input.shape)

    # Compute pairwise distances
    dist = torch.cdist(quant_input, embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
    # print(self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)).shape)
    # print(dist.shape)

    # Find index of nearest embedding
    min_encoding_indices = torch.argmin(dist, dim=-1)
    # print(min_encoding_indices.shape)

    # Select the embedding weights
    quant_out = torch.index_select(embedding.weight, 0, min_encoding_indices.view(-1))
    quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
    return quant_out

class VQVAETrainer(nn.Module):
    def __init__(self, in_c, out_c, down_sampling_times, encode_laten_channel, Z_size, load_model_path, lr=1e-4):
        super().__init__()
        self.vqvae = VQVAE(in_c=in_c,
                           out_c=out_c,
                           st_c=128,
                           down_sampling_times=down_sampling_times,
                           encode_laten_channel=encode_laten_channel,
                           Z_size=Z_size)
        self.vqvae = self.vqvae.to(device)
        embedding_weights = self.vqvae.embedding.weight.data
        max_weight = torch.max(embedding_weights)
        min_weight = torch.min(embedding_weights)
        print(f"Max weight: {max_weight}, Min weight: {min_weight}")
        self.optim = optim.Adam(self.vqvae.parameters(), lr=lr)
        self.scaler = amp.GradScaler()
        self.loss_fn = nn.MSELoss()
        # if torch.cuda.device_count() > 1:
        #     self.vqvae = nn.DataParallel(self.vqvae)
        if load_model_path:
            self.load(load_model_path)
        embedding_weights = self.vqvae.embedding.weight.data
        max_weight = torch.max(embedding_weights)
        min_weight = torch.min(embedding_weights)
        print(f"Max weight: {max_weight}, Min weight: {min_weight}")

    def train_model(self, train_loader, num_epochs):
        self.vqvae.train()
        for epoch in tqdm.tqdm(range(num_epochs)):
            loss_es = []
            for x, _ in tqdm.tqdm(train_loader):
                x = x.to(device)
                self.optim.zero_grad()
                with amp.autocast():
                    output, quantize_losses = self.vqvae(x)
                    loss = self.loss_fn(output, x) + quantize_losses
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                loss_es.append(loss.item())
            print(f"Epoch {epoch} Loss {sum(loss_es)/len(loss_es)}")
            self.save(f"model/checkpoint/VQVAE{epoch//20}.pth")
            show_img_VAE(x,output,epoch)

    def save(self, path):
        state_dict = {
            "vqvae": self.vqvae.state_dict(),
            "embedding":self.vqvae.embedding.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_rate": self.optim.param_groups[0]["lr"]
        }
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.vqvae.load_state_dict(state_dict["vqvae"])
        self.vqvae.embedding.load_state_dict(state_dict["embedding"])
        self.optim.load_state_dict(state_dict["optim"])
        self.scaler.load_state_dict(state_dict["scaler"])
        for param_group in self.optim.param_groups:
            param_group["lr"] = state_dict["lr_rate"]
    
    def inference(self, train_loader, names):
        self.vqvae.eval()
        for x, _ in train_loader:
            x = x.to(device)
            output, _ = self.vqvae(x)
            show_img_VAE(x, output, names)
            break
    

class diffusion_model:
    def __init__(self,in_c,out_c,st_channel,channel_multi,att_channel,embedding_time_dim,time_exp,num_head,d_model,num_resbox,allow_att,concat_up_down,concat_all_resbox,down_sampling_times,encode_laten_channel,Z_size,load_model_path,load_model_path_VQVAE,lr) -> None:
        self.model = UNet(
            in_c=encode_laten_channel,
            out_c=encode_laten_channel,
            st_channel=st_channel,
            channel_multi=channel_multi,
            att_channel=att_channel,
            embedding_time_dim=embedding_time_dim,
            time_exp=time_exp,
            num_head=num_head,
            d_model=d_model,
            num_resbox=num_resbox,
            allow_att=allow_att,
            concat_up_down=concat_up_down,
            concat_all_resbox=concat_all_resbox
        )
        # self.model = UNet(encode_laten_channel,encode_laten_channel,st_channel,channel_multi,att_channel,embedding_time_dim,time_exp,num_head,d_model,num_resbox,allow_att,concat_up_down,concat_all_resbox)
        self.vqvae = VQVAE(in_c=in_c,
                           out_c=out_c,
                           st_c=128,
                           down_sampling_times=down_sampling_times,
                           encode_laten_channel=encode_laten_channel,
                           Z_size=Z_size)
        self.model = self.model.to(device)
        self.vqvae = self.vqvae.to(device)
        self.down_sampling_times =down_sampling_times
        embedding_weights = self.vqvae.embedding.weight.data
        max_weight = torch.max(embedding_weights)
        min_weight = torch.min(embedding_weights)
        print(f"Max weight: {max_weight}, Min weight: {min_weight}")
        
        self.optim = optim.Adam(self.model.parameters(),lr=lr)
        # self.vqvae_optim = optim.Adam(self.vqvae.parameters(),lr=1e-6)
        self.scaler = amp.GradScaler()
        # self.vqvae_scaler = amp.GradScaler()
        self.loss = nn.MSELoss()
        # self.vqvae_loss = nn.MSELoss()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.vqvae = nn.DataParallel(self.vqvae)
        if load_model_path or load_model_path_VQVAE:
            self.load(load_model_path,load_model_path_VQVAE)
            print("load")
        self.embedding = self.vqvae.embedding
        embedding_weights = self.vqvae.embedding.weight.data
        max_weight = torch.max(embedding_weights)
        min_weight = torch.min(embedding_weights)
        print(f"Max weight: {max_weight}, Min weight: {min_weight}")
        embedding_weights = self.embedding.weight.data
        max_weight = torch.max(embedding_weights)
        min_weight = torch.min(embedding_weights)
        print(f"Max weight: {max_weight}, Min weight: {min_weight}")

    def train(self,train_loader,num_epoch):
        self.model.train()
        self.vqvae.eval()

        for param in self.vqvae.parameters():
            param.requires_grad = False  # No gradient updates for vqvae
            
        for epoch in tqdm.tqdm(range(num_epoch)):
            loss_es = []
            # loss_vqvae = []
            for i,(x,_) in enumerate(tqdm.tqdm(train_loader)):
                x = x.to(device)
                t = torch.randint(0, 1000, (x.size(0),), device=device).long()
                self.optim.zero_grad()
                # self.vqvae_optim.zero_grad()
                with amp.autocast():
                    # output,quantize_losses = self.vqvae(x)
                    xx = self.vqvae.input_layer(x)
                    for enc in self.vqvae.encode:
                        xx = enc(xx)
                    xx = self.vqvae.pre_quant_conv(xx)
                    quant_out = codebook(xx,self.embedding)
                    # quant_out = codebook(output,self.embedding)
                    # loss = self.loss(self.model(quant_out,t),x)
                    loss = get_loss(self.model,quant_out,t)
                    # vqvae_loss = self.vqvae_loss(output,x) + quantize_losses
                self.scaler.scale(loss).backward()
                # self.vqvae_scaler.scale(vqvae_loss).backward()
                self.scaler.step(self.optim)
                # self.vqvae_scaler.step(self.vqvae_optim)
                self.scaler.update()
                # self.vqvae_scaler.update()
                loss_es.append(loss.item())
                # loss_vqvae.append(vqvae_loss.item())
                # if i % 100 == 0:
            print(f"Epoch {epoch} Loss {sum(loss_es)/len(loss_es)}")
            self.save(f"model/checkpoint/DDPM_T_VQVAE{epoch//20}.pth")
            self.inference(epoch,x.size(2)//(2**self.down_sampling_times))

    def save(self,path):
        state_dict = {"model":self.model.state_dict(),
                    #   "vqvae":self.vqvae.state_dict(),
                      "optim":self.optim.state_dict(),
                    #   "vqvae_optim":self.vqvae_optim.state_dict(),
                    #   "embedding":self.embedding.state_dict(),
                      "scaler":self.scaler.state_dict(),
                    #   "vqvae_scaler":self.vqvae_scaler.state_dict(),
                      "lr_rate":self.optim.param_groups[0]["lr"]}
        torch.save(state_dict,path)


    def load(self,path,path_vqvae):
        if path:
            state_dict = torch.load(path)
            self.model.load_state_dict(state_dict["model"])
            # self.vqvae.load_state_dict(state_dict["vqvae"])
            self.optim.load_state_dict(state_dict["optim"])
            # self.vqvae_optim.load_state_dict(state_dict["vqvae_optim"])
            # self.embedding.load_state_dict(state_dict["embedding"])
            self.scaler.load_state_dict(state_dict["scaler"])
            # self.vqvae_scaler.load_state_dict(state_dict["vqvae_scaler"])
            for param_group in self.optim.param_groups:
                param_group["lr"] = state_dict["lr_rate"]
        if path_vqvae:
            state_dict_vqvae = torch.load(path_vqvae)
            self.vqvae.load_state_dict(state_dict_vqvae["vqvae"])
            # self.vqvae.embedding.load_state_dict(state_dict_vqvae["embedding"])
            

    def inference(self,names,size):
        self.model.eval()
        sample_plot_image(self.vqvae,self.model,names,size)


class diffusion_model_No_VQVAE:

    def __init__(self, in_c, out_c, st_channel, channel_multi, att_channel, embedding_time_dim, time_exp, num_head, d_model, num_resbox, allow_att, concat_up_down, concat_all_resbox, load_model_path):
        self.model = UNet(in_c, out_c, st_channel, channel_multi, att_channel, embedding_time_dim, time_exp, num_head, d_model, num_resbox, allow_att, concat_up_down, concat_all_resbox)
        self.model = self.model.to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=5e-4)
        self.scaler = amp.GradScaler()
        self.loss = nn.MSELoss()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        if load_model_path:
            self.load(load_model_path)

    def train(self, train_loader, num_epoch):
        self.model.train()
        for epoch in tqdm.tqdm(range(num_epoch)):
            loss_es = []
            for i, (x, _) in enumerate(tqdm.tqdm(train_loader)):
                x = x.to(device)
                t = torch.randint(0, 1000, (x.size(0),), device=device).long()
                self.optim.zero_grad()
                with amp.autocast():
                    loss = get_loss(self.model, x, t)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                loss_es.append(loss.item())
            print(f"Epoch {epoch} Loss {sum(loss_es)/len(loss_es)}")
            self.save(f"model/checkpoint/DDPM_T{epoch//20}.pth")
            self.inference(epoch,32)

    def save(self, path):
        state_dict = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_rate": self.optim.param_groups[0]["lr"]
        }
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])
        self.scaler.load_state_dict(state_dict["scaler"])
        for param_group in self.optim.param_groups:
            param_group["lr"] = state_dict["lr_rate"]
            
    def inference(self, names, size):
        self.model.eval()
        # sample_plot_image(None, self.model, names)
        sample_plot_image_no_VQVAE(self.model,names,size)