import torch.nn as nn
import torch
from vae import VAE
from ddpm import DDPM


class LDM(nn.Module):
    def __init__(self, diffusion_depth, feature_channels : int = 1, label_dim : int = 1, conv_map : dict = None, precision=torch.float16):
        super(LDM, self).__init__()
        
        self.enc_dec = VAE(num_channels=feature_channels)
        self.z_min = nn.Parameter(torch.zeros(self.enc_dec.latent_shape, dtype=precision), requires_grad=False)
        self.z_max = nn.Parameter(torch.zeros(self.enc_dec.latent_shape, dtype=precision), requires_grad=False)
        self.ddpm = DDPM(diffusion_depth=diffusion_depth, latent_shape=self.enc_dec.latent_shape, label_dim=label_dim, conv_map=conv_map)

    def noise(self, z, t):
        z_t, eps = self.ddpm.noise(z, t)
        return z_t, eps
    
    def predict_noise(self, z_i, t, y):
        return self.ddpm.predict_noise(z_i, t, y)
    
    def set_latent_range(self, z_min, z_max):
        self.z_min = nn.Parameter(z_min, requires_grad=False)
        self.z_max = nn.Parameter(z_max, requires_grad=False)
    
    def normalize_z(self, z):
        return (torch.div(z - self.z_min, self.z_max - self.z_min) * 2 - 1).to(dtype=z.dtype)
    
    def denormalize_z(self, z):
        return ((z + 1) / 2 * (self.z_max - self.z_min) + self.z_min).to(dtype=z.dtype)
    
    def forward(self, x, t, y):
        z = self.normalize_z(self.enc_dec.encode(x))
        z_i, eps = self.noise(z, t)
        pred_eps = self.predict_noise(z_i, t, y) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
        return eps, pred_eps