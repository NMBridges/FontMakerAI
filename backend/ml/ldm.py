import torch.nn as nn
import torch
from ml.vae import CNN_VAE, ImageProjector_VAE
from ml.ddpm import DDPM
import numpy as np
from tqdm import tqdm


class LDM(nn.Module):
    def __init__(self, **kwargs : dict):
        super(LDM, self).__init__()

        diffusion_depth = kwargs['diffusion_depth'] if 'diffusion_depth' in kwargs else 1000
        feature_dim = kwargs['feature_dim'] if 'feature_dim' in kwargs else 1
        label_dim = kwargs['label_dim'] if 'label_dim' in kwargs else 1
        embedding_dim = kwargs['embedding_dim'] if 'embedding_dim' in kwargs else 2048
        num_glyphs = kwargs['num_glyphs'] if 'num_glyphs' in kwargs else 26
        conv_map = kwargs['conv_map'] if 'conv_map' in kwargs else None
        precision = kwargs['precision'] if 'precision' in kwargs else torch.float32
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 6
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 32
        cond_dim = kwargs['cond_dim'] if 'cond_dim' in kwargs else 128

        # self.enc_dec = CNN_VAE(feature_dim=feature_dim, latent_dim=embedding_dim)
        self.enc_dec = ImageProjector_VAE(embedding_dim=embedding_dim, num_glyphs=num_glyphs)
        # self.ddpm = DDPM(diffusion_depth=diffusion_depth, latent_shape=self.enc_dec.latent_shape, label_dim=label_dim, conv_map=conv_map)
        self.ddpm = DDPM(diffusion_depth=diffusion_depth, label_dim=label_dim, num_layers=num_layers, embedding_dim=embedding_dim, num_glyphs=num_glyphs, num_heads=num_heads, cond_dim=cond_dim)

    def noise(self, z, t):
        z_t, eps = self.ddpm.noise(z, t)
        return z_t, eps
    
    def predict_noise(self, z_i, t, y):
        return self.ddpm.predict_noise(z_i, t, y)
    
    def denoise(self, z_t, t, y, cfg_coeff=3):
        return self.ddpm.denoise(z_t, t, y, cfg_coeff=cfg_coeff)
    
    def normalize_z(self, z):
        return (torch.div(z - self.enc_dec.z_min, self.enc_dec.z_max - self.enc_dec.z_min) * 2 - 1).to(dtype=z.dtype)
    
    def denormalize_z(self, z):
        return ((z + 1) / 2 * (self.enc_dec.z_max - self.enc_dec.z_min) + self.enc_dec.z_min).to(dtype=z.dtype)
    
    def feature_to_latent(self, x):
        return self.normalize_z(self.enc_dec.encode(x))
    
    def latent_to_feature(self, z):
        return self.enc_dec.decode(self.denormalize_z(z))
    
    def forward(self, x, t, y):
        z = self.feature_to_latent(x)
        z_i, eps = self.noise(z, t)
        pred_eps = self.predict_noise(z_i, t, y) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
        return eps, pred_eps
        # z_i, eps = self.noise(x, t)
        # pred_eps = self.predict_noise(z_i, t, y) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
        # return eps, pred_eps
    
    @torch.no_grad()
    def sample(self, latent_shape, label=None, cfg_coeff=3, device='cuda', precision=torch.float32):
        diff_timestep = self.ddpm.alphas.shape[0] - 1
        times = torch.IntTensor(np.linspace(0, diff_timestep, (diff_timestep+1) // 4, dtype=int)).to(device)
        z = torch.randn(latent_shape).to(device, dtype=precision)
        for i in tqdm(range(diff_timestep, 0, -1), desc='Sampling...'):
            z = self.denoise(z, times[i:i+1], label, cfg_coeff=cfg_coeff)
        return self.latent_to_feature(z)