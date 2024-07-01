import torch
import torch.nn as nn
import numpy as np


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mhsa_out, _ = self.MHA(x, x, x)
        x = self.norm_1(mhsa_out + x)
        ff_out = self.ff(x)
        x = self.norm_2(ff_out + x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, dropout_rate : float, alphas : torch.Tensor,
                    alpha_bars : torch.Tensor, device : torch.device) -> nn.Module:
        super(TransformerEncoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout_rate)

        d = 100 # Dimension of time embedding
        self.embedded_frequencies = torch.Tensor(torch.pow(torch.Tensor([0.0001]), 2 / d * torch.ceil(torch.linspace(1, d, d) / 2))).to(device)
        self.sin_hot = (torch.linspace(1, d, d) % 2 == 0).to(device)
        self.cos_hot = (torch.linspace(1, d, d) % 2 == 1).to(device)
        self.time_map = nn.Linear(d, embedding_dim)

        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.transformer_encoder_layers = nn.Sequential(
            *[TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        )

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                param.bias.data.fill_(0.01)
        self.transformer_encoder_layers.apply(init_weights)
    
    def time_embedding(self, t : torch.Tensor):
        # sine embedding
        return torch.sin(torch.outer(t, self.embedded_frequencies)) * self.sin_hot + torch.cos(torch.outer(t, self.embedded_frequencies)) * self.cos_hot

    def noise_prediction(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x_t (torch.Tensor): the embedded source sequence to predict
        t (torch.Tensor): the timestep for the diffusion process

        Returns:
        --------
        torch.Tensor: the encoded sequence (batch_size, seq_len, embedding_dim)
        '''
        # x : (batch_size, seq_len, embedding_dim)
        time_embedding = self.time_map(self.time_embedding(t))[:,None,:]
        embeddings = torch.cat([time_embedding, x_t], dim=1)
        embeddings += self.pos_embed[:,:embeddings.shape[1],:]
        return self.transformer_encoder_layers(self.dropout(embeddings))[:,1:,:]

    def forward(self, x_t : torch.Tensor, t : torch.Tensor):
        # x : (batch_size, seq_len, embedding_dim)
        pred_noise = self.noise_prediction(x_t, t)
        mean = 1 / torch.sqrt(self.alphas[t])[:,None,None] * (x_t - (1 - self.alphas[t])[:,None,None] / torch.sqrt(1 - self.alpha_bars[t])[:,None,None] * pred_noise)
        var = ((1 - self.alphas[t]) * (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]))[:,None,None]
        eps = torch.randn_like(mean).to(self.device) * (t > 1)[:,None,None]
        return mean + torch.sqrt(var) * eps


class FontDiffusionModel(nn.Module):
    def __init__(self, depth : int, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, embedder : nn.Module = None, dropout_rate : float = 0.1,
                    device : torch.device = torch.device('cpu')):
        super(FontDiffusionModel, self).__init__()

        self.device = device
        self.depth = depth
        self.alphas = torch.linspace(0.9999, 0.98, depth+1).to(device)
        self.alpha_bars = torch.Tensor([torch.prod(self.alphas[:i+1]) for i in range(depth+1)]).to(device)
        
        self.decoder = TransformerEncoder(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            alphas=self.alphas,
            alpha_bars=self.alpha_bars,
            device=device
        ).to(device)

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder
        if embedder is None:
            self.embedder = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True).to(device)
        self.token_space = nn.Linear(embedding_dim, vocab_size).to(device)
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x_0 : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        # x_0 : (batch_size, seq_len, embedding_dim)
        mean = torch.sqrt(self.alpha_bars[t])[:,None,None] * x_0 # (batch_size, seq_len, embedding_dim)
        var = (1 - self.alpha_bars[t])[:,None,None]
        eps = torch.randn_like(mean).to(self.device)
        return mean + torch.sqrt(var) * eps, eps # x_t, eps

    def decode(self, x_t : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        return self.decoder(x_t, t)

    def sample(self, size : torch.Size) -> torch.Tensor:
        # size : (batch_size, seq_len)
        x_T = torch.randn(size).to(self.device)
        times = torch.IntTensor(torch.linspace(0, self.depth, self.depth + 1, dtype=torch.int)).to(self.device)

        # NOTE: model should be in eval mode
        t = self.depth
        traj = [x_T]
        while t >= 1:
            x_T = self.decode(x_T, times[t:t+1])
            t -= 1
            traj.append(x_T)
        return x_T, traj
