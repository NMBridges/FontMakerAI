import torch
import torch.nn as nn
import numpy as np
from attention import MultiheadCrossAttention

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
        # self.cond_map = nn.Sequential(
        #     nn.Linear(cond_dimension, out_channels, bias=False),
        #     nn.LeakyReLU(0.2)
        # )
        self.cond_map = MultiheadCrossAttention(out_channels, cond_dimension, 8)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, time_embedding, y):
        dc_new = self.conv2(self.cond_map(self.conv1(x), y) + self.time_map(time_embedding)[:,:,None,None])
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

        factor1 = 16
        factor2 = 2
        factor3 = 1
        # self.down1 = UNetDownBlock(in_channels, 64 * factor1, time_dimension, cond_dimension, conv_map)
        # self.down2 = UNetDownBlock(64 * factor1, 128 * factor2, time_dimension, cond_dimension, conv_map)
        # self.down3 = UNetDownBlock(128 * factor2, 256 * factor2, time_dimension, cond_dimension, conv_map)
        # self.down4 = UNetDownBlock(256 * factor2, 512 * factor2, time_dimension, cond_dimension, conv_map)
        # self.down5 = UNetDownBlock(512 * factor2, 1024 * factor3, time_dimension, cond_dimension, conv_map)
        # self.down6 = UNetDownBlock(1024 * factor3, 2048 * factor3, time_dimension, cond_dimension, conv_map)
        # self.layer7 = UNetDoubleConv(2048 * factor3, 2048 * factor3, time_dimension, cond_dimension, conv_map)
        # self.up6 = UNetUpBlock(2048 * factor3, 1024 * factor3, time_dimension, cond_dimension, conv_map)
        # self.up5 = UNetUpBlock(1024 * factor3, 512 * factor2, time_dimension, cond_dimension, conv_map)
        # self.up4 = UNetUpBlock(512 * factor2, 256 * factor2, time_dimension, cond_dimension, conv_map)
        # self.up3 = UNetUpBlock(256 * factor2, 128 * factor2, time_dimension, cond_dimension, conv_map)
        # self.up2 = UNetUpBlock(128 * factor2, 64 * factor1, time_dimension, cond_dimension, conv_map)
        # self.up1 = UNetUpBlock(64 * factor1, in_channels, time_dimension, cond_dimension, conv_map)

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