import torch
import torch.nn as nn
import numpy as np
from backend.ml.backbones.dit import MultiheadCrossAttention


class UNetTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, time_dim, cond_dim, num_heads):
        super(UNetTransformerLayer, self).__init__()

        self.cond_proj = nn.Sequential(
            nn.Linear(time_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 9 * embedding_dim)
        )
        self.MHA = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.conditioning = MultiheadCrossAttention(embedding_dim, cond_dim, num_heads)
        self.ff = nn.Linear(embedding_dim, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 8*8, embedding_dim))

    def scale_shift(self, x, scale, shift):
        return x * (1 + scale) + shift

    def forward(self, x, time_embedding, y):
        '''
        Parameters:
        -----------
        x (torch.Tensor): (bs, channels, x_1, x_2)
        '''
        xs = x.shape
        
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        adaln_z = self.cond_proj(time_embedding).unflatten(1, (9, -1)) # (bs, 9, channels)
        
        x_ = self.scale_shift(self.norm1(x), adaln_z[:,0:1], adaln_z[:,1:2])
        qk_x_ = x_ + self.pos_emb[:,:x_.shape[1],:]
        x = self.scale_shift(self.MHA(qk_x_, qk_x_, x_)[0], adaln_z[:,2:3], x)
    
        if y is not None:
            x_ = self.scale_shift(self.norm2(x), adaln_z[:,3:4], adaln_z[:,4:5])
            x = self.scale_shift(self.conditioning(x_, y), adaln_z[:,5:6], x)

        x_ = self.scale_shift(self.norm3(x), adaln_z[:,6:7], adaln_z[:,7:8])
        x = self.scale_shift(self.ff(x_), adaln_z[:,8:9], x)
        
        x = x.unflatten(1, xs[2:4]).permute(0, 3, 1, 2)
        return x

class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetDoubleConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2)
        )
        self.time_map = nn.Sequential(
            nn.Linear(time_dimension, out_channels, bias=False),
            nn.LeakyReLU(0.2)
        )
        # self.attn = UNetTransformerLayer(embedding_dim=out_channels, time_dim=time_dimension, cond_dim=cond_dimension, num_heads=8)
        self.cond_map = MultiheadCrossAttention(out_channels, cond_dimension, 8)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, time_embedding, y):
        dc_new = self.conv2(self.cond_map(self.conv1(x), y) + self.time_map(time_embedding)[:,:,None,None])
        # dc_new = self.attn(dc_new, time_embedding, y)
        return self.batch_norm(self.act(dc_new + self.res_conv(x)))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetDownBlock, self).__init__()

        self.double_conv = UNetDoubleConv(in_channels, out_channels, time_dimension, cond_dimension, conv_map)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x, time_embedding, y):
        conved = self.double_conv(x, time_embedding, y)
        return conved, self.pool(conved)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetUpBlock, self).__init__()

        self.unpool = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        self.double_conv = UNetDoubleConv(2 * in_channels, out_channels, time_dimension, cond_dimension, conv_map)

        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x, x_prior, time_embedding, y):
        return self.double_conv(torch.cat((x_prior, self.unpool(x)), dim=1), time_embedding, y)


class UNet(nn.Module):
    def __init__(self, in_channels, time_dimension, cond_dimension, conv_map):
        super(UNet, self).__init__()

        self.down1 = UNetDownBlock(in_channels, 512, time_dimension, cond_dimension, conv_map)
        self.down2 = UNetDownBlock(512, 1024, time_dimension, cond_dimension, conv_map)
        self.down3 = UNetDownBlock(1024, 2048, time_dimension, cond_dimension, conv_map)
        # self.down4 = UNetDownBlock(256 * factor2, 512 * factor2, time_dimension, cond_dimension, conv_map)
        # self.down5 = UNetDownBlock(512 * factor2, 1024 * factor3, time_dimension, cond_dimension, conv_map)
        # self.down6 = UNetDownBlock(1024 * factor3, 2048 * factor3, time_dimension, cond_dimension, conv_map)
        self.layer7 = UNetDoubleConv(2048, 2048, time_dimension, cond_dimension, conv_map)
        # self.up6 = UNetUpBlock(2048 * factor3, 1024 * factor3, time_dimension, cond_dimension, conv_map)
        # self.up5 = UNetUpBlock(1024 * factor3, 512 * factor2, time_dimension, cond_dimension, conv_map)
        # self.up4 = UNetUpBlock(512 * factor2, 256 * factor2, time_dimension, cond_dimension, conv_map)
        self.up3 = UNetUpBlock(2048, 1024, time_dimension, cond_dimension, conv_map)
        self.up2 = UNetUpBlock(1024, 512, time_dimension, cond_dimension, conv_map)
        self.up1 = UNetUpBlock(512, in_channels, time_dimension, cond_dimension, conv_map)

        self.last = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            
    def forward(self, x, time_embedding, y):
        x1, x_run = self.down1(x, time_embedding, y)
        x2, x_run = self.down2(x_run, time_embedding, y)
        x3, x_run = self.down3(x_run, time_embedding, y)
        # x4, x_run = self.down4(x_run, time_embedding, y)
        # x5, x_run = self.down5(x_run, time_embedding, y)
        # x6, x_run = self.down6(x_run, time_embedding, y)
        x_run = self.layer7(x_run, time_embedding, y)
        # x_run = self.up6(x_run, x6, time_embedding, y)
        # x_run = self.up5(x_run, x5, time_embedding, y)
        # x_run = self.up4(x_run, x4, time_embedding, y)
        x_run = self.up3(x_run, x3, time_embedding, y)
        x_run = self.up2(x_run, x2, time_embedding, y)
        return self.last(self.up1(x_run, x1, time_embedding, y))