import torch
import torch.nn as nn
from config import device


class ResDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(ResDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding * dilation, dilation=dilation)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding * dilation, dilation=dilation)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.relu(self.batchnorm1(self.conv1(x)))
        x2 = self.relu(self.batchnorm2(self.conv2(x1)) + self.conv_res(x))
        return x2
    

class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResDownBlock, self).__init__()
        self.res_double_conv = ResDoubleConv(in_channels, out_channels, kernel_size, 1, dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.res_double_conv(x)
        x = self.pool(x)
        return x


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResUpBlock, self).__init__()
        self.unpool = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.res_double_conv = ResDoubleConv(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2, dilation=dilation)

    def forward(self, x):
        x = self.unpool(x)
        x = self.res_double_conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, num_channels : int):
        super(VAE, self).__init__()

        self.num_channels = num_channels
        self.latent_shape = (64, 8, 8)

        # (num_channels, 64, 64) -> (4, 4, 4)
        self.encoder = nn.Sequential(
            ResDownBlock(in_channels=self.num_channels, out_channels=128, kernel_size=3, dilation=1), # 32x32
            ResDownBlock(in_channels=128, out_channels=256, kernel_size=3, dilation=1), # 16x16
            ResDownBlock(in_channels=256, out_channels=self.latent_shape[0], kernel_size=3, dilation=1), # 8x8
        )

        self.mu_pred = nn.Sequential(
            ResDoubleConv(self.latent_shape[0], self.latent_shape[0], kernel_size=3, stride=1, padding=1, dilation=1), # 3x1x1
            nn.Conv2d(self.latent_shape[0], self.latent_shape[0], kernel_size=1, stride=1, padding=0)
        )
        self.logvar_pred = nn.Sequential(
            ResDoubleConv(self.latent_shape[0], self.latent_shape[0], kernel_size=3, stride=1, padding=1, dilation=1), # 3x1x1
            nn.Conv2d(self.latent_shape[0], self.latent_shape[0], kernel_size=1, stride=1, padding=0)
        )

        # (4, 4, 4) -> (num_channels, 64, 64)
        self.decoder = nn.Sequential(
            ResUpBlock(in_channels=self.latent_shape[0], out_channels=64, kernel_size=3, dilation=1), # 16x16
            ResUpBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=1), # 32x32
            ResUpBlock(in_channels=64, out_channels=self.num_channels, kernel_size=3, dilation=1), # 64x64
            nn.Conv2d(self.num_channels, self.num_channels, kernel_size=1, stride=1, padding=0, groups=self.num_channels)
        )

        for param in self.modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.ConvTranspose2d):
                if hasattr(param, 'weight'):
                    nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean, dtype=mean.dtype).to(device)
        return mean + torch.sqrt(logvar.exp()) * eps, eps
    
    def encode_params(self, x):
        base = self.encoder(x)
        mu = self.mu_pred(base)
        logvar = self.logvar_pred(base)
        return mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encode_params(x)
        return self.reparameterize(mu, logvar)[0]
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)[0]
        x_hat = self.decode(z)
        return x_hat, mu, logvar