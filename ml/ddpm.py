import torch
import torch.nn as nn
import numpy as np
from config import device
from unet import UNet


class DDPM(nn.Module):
    def __init__(self, diffusion_depth : int, latent_shape : tuple, label_dim : int, conv_map : dict):
        super(DDPM, self).__init__()
        
        self.alphas = nn.Parameter(torch.Tensor(np.linspace(0.9999, 0.98, diffusion_depth+1))[:,None,None,None], requires_grad=False)
        self.alpha_bars = nn.Parameter(torch.Tensor([torch.prod(self.alphas[:i+1]) for i in range(diffusion_depth+1)])[:,None,None,None], requires_grad=False)

        d = 128 # Dimension of time embedding; Dimension of condition embedding

        # Time embedding
        self.embedded_frequencies = nn.Parameter(torch.Tensor(np.power(np.array([0.0001]), 2 / d * np.ceil(np.linspace(1, d, d) / 2))), requires_grad=False)
        self.sin_hot = nn.Parameter(torch.Tensor(np.linspace(1, d, d) % 2 == 0), requires_grad=False)
        self.cos_hot = nn.Parameter(torch.Tensor(np.linspace(1, d, d) % 2 == 1), requires_grad=False)

        # Condition embedding
        self.cond_embedding = nn.Linear(label_dim, d)

        self.noise_pred = UNet(in_channels=latent_shape[0], time_dimension=d, cond_dimension=d, conv_map=conv_map).to(device)

    def reparameterize(self, mean, var):
        eps = torch.randn_like(mean).to(device)
        return mean + torch.sqrt(var) * eps, eps

    def noise(self, x0, t):
        # mean, var for x_t sampled along q(x_t | x_0)
        mean = torch.sqrt(self.alpha_bars[t]) * x0
        var = (1 - self.alpha_bars[t])
        x_t, eps = self.reparameterize(mean, var)
        return x_t, eps
        
    def time_embedding(self, t):
        # sine embedding
        return torch.sin(torch.outer(t, self.embedded_frequencies)) * self.sin_hot + torch.cos(torch.outer(t, self.embedded_frequencies)) * self.cos_hot

    def denoise(self, x_t, t, y, cfg_coeff=3):
        predicted_noise = self.predict_noise(x_t, t, y)
        if cfg_coeff > 0:
            unconditional_predicted_noise = self.predict_noise(x_t, t, None)
            predicted_noise = torch.lerp(predicted_noise, unconditional_predicted_noise, -cfg_coeff)

        mean = 1 / torch.sqrt(self.alphas[t]) * (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * predicted_noise)
        var = ((1 - self.alphas[t]) * (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]))
        eps = torch.randn_like(mean).to(device) * (t > 1)[:,None,None,None]
        return mean + torch.sqrt(var) * eps

    def x0_pred(self, x_t, t, y):
        predicted_noise = self.predict_noise(x_t, t, y)
        x0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) * predicted_noise) / torch.sqrt(self.alpha_bars[t])
        return x0

    def predict_noise(self, x_t, t, y):
        time_emb = self.time_embedding(t)
        if y is None:
            cond_emb = None
        else:
            cond_emb = self.cond_embedding(y)
        predicted_noise = self.noise_pred(x_t, time_emb, cond_emb)
        
        return predicted_noise
    
    def forward(self, x, t, y):
        x_i, eps = self.noise(x, t) # x_{i}, eps_true ~ q(x_{i} | x_{0})
        pred_eps = self.predict_noise(x_i, t, y) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
        return eps, pred_eps