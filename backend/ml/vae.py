import torch
import torch.nn as nn
import numpy as np
from backend.config import device


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
    def __init__(self):
        super(VAE, self).__init__()

        self.latent_shape = None

        # (num_channels, 64, 64) -> (4, 4, 4)
        self.encoder = None

        self.mu_pred = None
        self.logvar_pred = None

        # (4, 4, 4) -> (num_channels, 64, 64)
        self.decoder = None

        for param in self.modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.ConvTranspose2d):
                if hasattr(param, 'weight'):
                    nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)

    def base_forward(self, x):
        raise NotImplementedError
    
    def base_backward(self, x):
        raise NotImplementedError

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean, dtype=mean.dtype).to(device)
        return mean + torch.sqrt(logvar.exp()) * eps, eps

    def param_forward(self, x):
        '''
        x (torch.Tensor): (bs, num_glyphs, 128, 128)
        returns (torch.Tensor): (bs, num_glyphs, embedding_dim)
        '''
        base = self.base_forward(x) # (bs, num_glyphs, embedding_dim)
        mu = self.mu_pred(base)
        logvar = self.logvar_pred(base)
        return mu, logvar
    
    def encode(self, x):
        mu, logvar = self.param_forward(x)
        return self.reparameterize(mu, logvar)[0]
    
    def decode(self, x):
        '''
        x (torch.Tensor): (bs, num_glyphs, embedding_dim * 2)
        returns (torch.Tensor): (bs, num_glyphs, 128, 128)
        '''
        return self.base_backward(x)
    
    def forward(self, x):
        mu, logvar = self.param_forward(x)
        z = self.reparameterize(mu, logvar)[0]
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    

class CNN_VAE(VAE):
    def __init__(self, feature_dim : int, latent_dim : int, compression_rate : int = 8):
        super(CNN_VAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        assert np.log2(compression_rate).is_integer(), "Compression rate must be a power of 2"
        self.compression_rate = compression_rate
        self.num_compressions = int(np.log2(compression_rate))
        self.layerwise_channels = [self.feature_dim] + [128] * self.num_compressions + [self.latent_dim]
        
        self.encoder = nn.Sequential(
            ResDoubleConv(self.layerwise_channels[0], self.layerwise_channels[1], kernel_size=3, stride=1, padding=1, dilation=1),
            *[ResDownBlock(in_channels=self.layerwise_channels[i+1], out_channels=self.layerwise_channels[i+2], kernel_size=3, dilation=1) for i in range(self.num_compressions)]
        )
        self.mu_pred = nn.Sequential(
            ResDoubleConv(self.layerwise_channels[-1], self.layerwise_channels[-1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv3d(self.layerwise_channels[-1], self.layerwise_channels[-1], kernel_size=1, stride=1, padding=0)
        )
        self.logvar_pred = nn.Sequential(
            ResDoubleConv(self.layerwise_channels[-1], self.layerwise_channels[-1], kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv3d(self.layerwise_channels[-1], self.layerwise_channels[-1], kernel_size=1, stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            ResDoubleConv(self.layerwise_channels[-1], self.layerwise_channels[-2], kernel_size=3, stride=1, padding=1, dilation=1),
            *[ResUpBlock(in_channels=self.layerwise_channels[i], out_channels=self.layerwise_channels[i-1], kernel_size=3, dilation=1) for i in range(self.num_compressions, 0, -1)],
            nn.Conv3d(self.layerwise_channels[0], self.layerwise_channels[0], kernel_size=1, stride=1, padding=0, groups=self.layerwise_channels[0])
        )

        self.z_min = nn.Parameter(-torch.ones(self.latent_dim,), requires_grad=False)
        self.z_max = nn.Parameter(torch.ones(self.latent_dim,), requires_grad=False)

        for param in self.modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.ConvTranspose2d):
                if hasattr(param, 'weight'):
                    nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)
    
    def base_forward(self, x):
        return self.encoder(x)
    
    def base_backward(self, x):
        return self.decoder(x)
    

class ImageProjector_VAE(VAE):
    def __init__(self, embedding_dim : int, num_glyphs : int, dropout_rate : float = 0.2):
        super(ImageProjector_VAE, self).__init__()
        
        self.latent_shape = (1, num_glyphs, embedding_dim)
        self.num_glyphs = num_glyphs
        self.net = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=8, stride=8, padding=0), # 16x16
            nn.RMSNorm((256, 16, 16)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=0), # 4x4
            nn.RMSNorm((512, 4, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=0), # 2x2
            nn.RMSNorm((1024, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, embedding_dim, kernel_size=2, stride=1, padding=0), # 1x1
            nn.RMSNorm((embedding_dim, 1, 1)),
            nn.LeakyReLU(0.2)
        )
        self.mu_pred = nn.Linear(embedding_dim, embedding_dim)
        self.logvar_pred = nn.Linear(embedding_dim, embedding_dim)

        self.inv_net = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 1024, kernel_size=2, stride=1, padding=0), # 2x2
            nn.RMSNorm((1024, 2, 2)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0), # 4x4
            nn.RMSNorm((512, 4, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0), # 16x16
            nn.RMSNorm((256, 16, 16)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 1, kernel_size=8, stride=8, padding=0), # 128x128
            nn.Tanh()
        )

        self.z_min = nn.Parameter(-torch.ones(self.latent_dim,), requires_grad=False)
        self.z_max = nn.Parameter(torch.ones(self.latent_dim,), requires_grad=False)

        for param in self.modules():
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose2d) or isinstance(param, nn.ConvTranspose3d):
                if hasattr(param, 'weight'):
                    nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)

    def base_forward(self, x):
        '''
        x (torch.Tensor): (bs, num_glyphs, 128, 128)
        returns (torch.Tensor): (bs, num_glyphs, embedding_dim)
        '''
        return self.net(x.view(x.shape[0] * x.shape[1], 1, 128, 128)).view(x.shape[0], x.shape[1], -1)
    
    def base_backward(self, x):
        '''
        x (torch.Tensor): (bs, num_glyphs, embedding_dim)
        returns (torch.Tensor): (bs, num_glyphs, 128, 128)
        '''
        return self.inv_net(x.view(x.shape[0] * x.shape[1], -1, 1, 1)).view(x.shape[0], x.shape[1], 128, 128)