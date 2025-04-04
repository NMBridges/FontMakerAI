import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from config import device
from ml.fontmodel import LearnedAbsolutePositionalEmbedding


class MultiheadCrossAttention(nn.Module):
    def __init__(self, channels, cond_dimension, num_heads, dropout_rate : float = 0.0):
        super(MultiheadCrossAttention, self).__init__()

        assert cond_dimension % num_heads == 0, "Projected dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_size = cond_dimension // self.num_heads
        self.q_proj = nn.Linear(channels, self.num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(cond_dimension, self.num_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(cond_dimension, self.num_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_size, channels, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x, c):
        ''' x: Tensor of shape (batch_size, seq_len, d)
            c: Tensor of shape (batch_size, num_conditions, condition_dimension)
        '''
        if c is None or c.shape[1] == 0:
            return x
        q = self.q_proj(x).unflatten(-1, (self.num_heads, self.head_size)).permute(0, 2, 1, 3) # (batch, nh, seq_len, hs)
        k = self.k_proj(c).unflatten(-1, (self.num_heads, self.head_size)).permute(0, 2, 1, 3) # (batch, nh, num_cond, hs)
        v = self.v_proj(c).unflatten(-1, (self.num_heads, self.head_size)).permute(0, 2, 1, 3) # (batch, nh, num_cond, hs)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_rate, is_causal=False)
        out_vals = self.out_proj(out.permute(0, 2, 1, 3).flatten(start_dim=-2)) # (bs, seq_len, d)
        return out_vals


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
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.res_double_conv(x)
        x = self.pool(x)
        return x
    

class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResUpBlock, self).__init__()
        self.unpool = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.res_double_conv = ResDoubleConv(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2, dilation=dilation)
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.unpool(x)
        x = self.res_double_conv(self.relu(x))
        return x
    

class SwiGLU_FNN(nn.Module):
    def __init__(self, embedding_dim : int, ff_dim : int):
        super(SwiGLU_FNN, self).__init__()

        self.linear_1 = nn.Linear(embedding_dim, ff_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, ff_dim, bias=False)
        self.linear_3 = nn.Linear(ff_dim, embedding_dim, bias=False)

    def forward(self, x : torch.Tensor):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x)
        return self.linear_3(nn.functional.silu(x1) * x2)


class DiTLayer(nn.Module):
    def __init__(self, embedding_dim : int, time_dim : int, cond_dim : int, num_heads : int):
        super(DiTLayer, self).__init__()
        
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            # nn.Dropout(0.2),
            nn.Linear(time_dim, 9 * embedding_dim)
        )
        self.MHSA = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.conditioning = MultiheadCrossAttention(embedding_dim, cond_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

        self.ff = SwiGLU_FNN(embedding_dim, 4 * embedding_dim)

    def scale_shift(self, x, scale, shift):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        '''
        x (torch.Tensor): (batch_size, sequence_length, d)
        c (torch.Tensor): (batch_size, num_conditions, condition_dimension)
            first condition is time embedding
        '''
        xs = x.shape # (bs, seq_len, d)

        # Strictly condition on time
        ayBayB = self.cond_proj(c[:,0]).unflatten(1, (9, -1)) # (bs, 9, d)

        # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        x_ = self.scale_shift(self.norm1(x), ayBayB[:,0:1], ayBayB[:,1:2])
        print(x_.shape, x.shape)
        x = self.scale_shift(self.MHSA(x_, x_, x_)[0], ayBayB[:,2:3], x)
        print(x.shape)
        
        # x_ = self.scale_shift(self.norm2(x), ayBayB[:,3:4], ayBayB[:,4:5])
        # x = self.scale_shift(self.conditioning(x_, c), ayBayB[:,5:6], x)

        x_ = self.scale_shift(self.norm3(x), ayBayB[:,6:7], ayBayB[:,7:8])
        print(x_.shape, x.shape)
        x = self.scale_shift(self.ff(x_), ayBayB[:,8:9], x)
        print(x.shape)
        return x


class DiT(nn.Module):
    def __init__(self, num_layers : int, embedding_dim : int, num_glyphs : int, num_heads : int, time_dim : int, cond_dim : int):
        super(DiT, self).__init__()

        self.pos_embed = LearnedAbsolutePositionalEmbedding(embedding_dim, num_glyphs)
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList([
            DiTLayer(embedding_dim=embedding_dim, time_dim=time_dim, cond_dim=cond_dim, num_heads=num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            # nn.Dropout(0.2),
            nn.Linear(time_dim, 2 * embedding_dim)
        )
        self.in_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(2 * embedding_dim, embedding_dim)

        for param in self.modules():
            if isinstance(param, nn.Linear):
                if hasattr(param, 'weight'):
                    # pass
                    # nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                    nn.init.xavier_normal_(param.weight)
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Conv3d) or isinstance(param, nn.ConvTranspose2d) or isinstance(param, nn.ConvTranspose3d):
                if hasattr(param, 'weight'):
                    # pass
                    nn.init.kaiming_normal_(param.weight, a=0.2, nonlinearity='leaky_relu')
                if hasattr(param, 'bias') and param.bias is not None:
                    nn.init.zeros_(param.bias)

    def forward(self, x, t, y):
        '''
        x (torch.Tensor): (batch_size, num_glyphs, embedding_dim)
        t (torch.Tensor): (batch_size, condition_dimension)
        y (torch.Tensor): (batch_size, condition_dimension)
        '''
        if y is None:
            c = t.unsqueeze(1)
        else:
            c = torch.cat((t.unsqueeze(1), y.unsqueeze(1)), dim=1)
        xs = x.shape # (bs, num_glyphs, embedding_dim)

        # Bring to embedding space
        x = self.in_proj(x)
        x_ = x
        x = self.pos_embed(x) # (bs, num_glyphs, embedding_dim)
        # x = self.dropout(x)

        # Layers
        for layer in self.layers:
            x = layer(x, c)

        # Layer norm; channel-wise conditional scale and shift
        yB = self.cond_proj(c[:,0]).unflatten(1, (2, -1))
        x = self.norm(x) * (1 + yB[:,0:1]) + yB[:,1:2]

        # # Residual connection
        print(x_.shape, x.shape)
        x = torch.cat((x_, x), dim=-1)
        x = self.out_proj(x)

        return x