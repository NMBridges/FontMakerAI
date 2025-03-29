import torch
import torch.nn as nn
import numpy as np


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int = 8):
        super(MultiheadSelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.q_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_size, embedding_dim, bias=False)
        self.scale = self.head_size**-0.5

    def forward(self, x):
        # x (bs, seq_len*z2*z3, z1)

        q = self.q_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len*z2*z3, z1) -> (bs, num_heads, seq_len*z2*z3, head_size)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 3, 1) # (bs, seq_len*z2*z3, z1) -> (bs, num_heads, head_size, seq_len*z2*z3)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len*z2*z3, z1) -> (bs, num_heads, seq_len*z2*z3, head_size)

        att_mat_pre = (q @ k) * self.scale # (bs, num_heads, seq_len*z2*z3, seq_len*z2*z3)
        # Since our linear projection has no bias, any zero vector will still be mapped to a zero vector and we can care about the projected k rather than raw x
        att_mat = att_mat_pre.softmax(dim=-1) # (bs, num_heads, seq_len*z2*z3, seq_len*z2*z3)
        att_vals = (att_mat @ v).permute(0, 2, 1, 3).flatten(start_dim=-2) # (bs, seq_len*z2*z3, num_heads * head_size)
        out_vals = self.out_proj(att_vals) # (bs, seq_len*z2*z3, z1)
        return out_vals


class MultiheadCrossAttention(nn.Module):
    def __init__(self, channels, cond_dimension, num_heads):
        super(MultiheadCrossAttention, self).__init__()

        assert cond_dimension % num_heads == 0, "Projected dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_size = cond_dimension // self.num_heads
        self.q_proj = nn.Linear(channels, self.num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(cond_dimension, self.num_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(cond_dimension, self.num_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_size, channels, bias=False)
        self.scale = self.head_size**-0.5
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, c):
        ''' x: Tensor of shape (batch_size, ..., z1)
            c: Tensor of shape (batch_size, num_conditions, condition_dimension)
        '''
        if c is None:
            return x
        q = self.q_proj(x).flatten(1, -2).unflatten(-1, (self.num_heads, self.head_size)).permute(0, 2, 1, 3) # (batch, nh, ..., hs)
        k = self.k_proj(c).unflatten(-1, (self.head_size, self.num_heads)).permute(0, 3, 2, 1) # (batch, nh, hs, num_cond)
        v = self.v_proj(c).unflatten(-1, (self.head_size, self.num_heads)).permute(0, 3, 1, 2) # (batch, nh, num_cond, hs)
        att_mask = (c == 0).all(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, self.num_heads, 1, 1) # (bs, num_heads, 1, num_cond)
        att_mat_pre = (q @ k) * self.scale # (batch, nh, ..., num_cond)
        att_mat_pre.masked_fill_(att_mask, -np.inf)
        att_mat = att_mat_pre.softmax(dim=-1) # (batch, nh, ..., num_cond)
        att_vals = (att_mat @ v).permute(0, 2, 1, 3) # (batch, nh, ..., num_cond) @ (batch, nh, num_cond, hs) -> (bs, ..., nh, hs)
        out_vals = self.out_proj(att_vals.flatten(start_dim=-2)).unflatten(1, x.shape[1:-1]) # (bs, ..., z1)
        return out_vals