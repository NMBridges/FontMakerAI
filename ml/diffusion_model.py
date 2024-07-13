import torch
import torch.nn as nn
import numpy as np
from unet import UNet


class VDM_Encoder(nn.Module):
    def __init__(self, depth : int, device : torch.device):
        super(VDM_Encoder, self).__init__()

        self.device = device
        self.alphas = torch.Tensor(np.linspace(0.9999, 0.98, depth+1)).to(device)
        self.alpha_bars = torch.Tensor([torch.prod(self.alphas[:i+1]) for i in range(depth+1)]).to(device)

    def reparameterize(self, mean, var):
        eps = torch.randn_like(mean).to(self.device)
        return mean + torch.sqrt(var) * eps, eps

    def forward(self, x0, t):
        # mean, var for x_t sampled along q(x_t | x_0)
        mean = torch.sqrt(self.alpha_bars[t])[:,None,None,None] * x0
        var = (1 - self.alpha_bars[t])[:,None,None,None]
        x_t, eps = self.reparameterize(mean, var)
        return x_t, eps


class VDM_Decoder(nn.Module):
    def __init__(self, depth : int, num_channels : int = 1, label_dim : int = 1, num_classes : int = None,
                    conv_map : dict = None, device : torch.device = None):
        super(VDM_Decoder, self).__init__()

        self.device = device
        self.alphas = torch.Tensor(np.linspace(0.9999, 0.98, depth+1)).to(device)
        self.alpha_bars = torch.Tensor([torch.prod(self.alphas[:i+1]) for i in range(depth+1)]).to(device)

        d = 100 # Dimension of time embedding
        self.embedded_frequencies = torch.Tensor(np.power(np.array([0.0001]), 2 / d * np.ceil(np.linspace(1, d, d) / 2))).to(device)
        self.sin_hot = torch.Tensor(np.linspace(1, d, d) % 2 == 0).to(device)
        self.cos_hot = torch.Tensor(np.linspace(1, d, d) % 2 == 1).to(device)

        c = 10 # Dimension of condition embedding
        self.num_classes = num_classes
        if num_classes is None:
            self.cond_embedding = nn.Sequential(
                nn.Linear(label_dim, c),
                nn.LeakyReLU(0.2),
                nn.Linear(c, c)
            )
        else:
            self.cond_embedding = nn.Embedding(num_classes, c)

        self.noise_pred = UNet(num_channels, d, c, conv_map).to(device)

    def time_embedding(self, t):
        # sine embedding
        return torch.sin(torch.outer(t, self.embedded_frequencies)) * self.sin_hot + torch.cos(torch.outer(t, self.embedded_frequencies)) * self.cos_hot

    def sample(self, x_t, t, y, cfg_coeff=3):
        # x_{t-1} sampled along p(x_{t-1} | x_t)
        predicted_noise = self.forward(x_t, t, y)
        if cfg_coeff > 0:
            unconditional_predicted_noise = self.forward(x_t, t, None)
            predicted_noise = torch.lerp(predicted_noise, unconditional_predicted_noise, -cfg_coeff)

        # DDPM
        mean = 1 / torch.sqrt(self.alphas[t])[:,None,None,None] * (x_t - (1 - self.alphas[t])[:,None,None,None] / torch.sqrt(1 - self.alpha_bars[t])[:,None,None,None] * predicted_noise)
        var = ((1 - self.alphas[t]) * (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]))[:,None,None,None]
        eps = torch.randn_like(mean).to(self.device) * (t > 1)
        return mean + torch.sqrt(var) * eps

        # Score-matching
        # eps = randn_like(x_t).to(device) * (t > 1)
        # # c = 0.01
        # return x_t + predicted_noise * c / 2 + np.sqrt(c) * eps

    def forward(self, x_t, t, y):
        time = self.time_embedding(t)
        if y is None:
            cond_emb = None
        elif self.num_classes is None:
            cond_emb = self.cond_embedding(y)
        else:
            cond_emb = self.cond_embedding(y.long())
        predicted_noise = self.noise_pred(x_t, time, cond_emb)

        return predicted_noise