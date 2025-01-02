import torch
import torch.nn as nn
import numpy as np
from tokenizer import Tokenizer


class OpVAE(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim : int, num_layers : int, hidden_dim : int, bidirectional : bool = True):
        super(OpVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)

        self.mu_pred = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.logvar_pred = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        self.ff = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x : torch.Tensor):
        out, (h, c) = self.encoder(x)
        new_h = h.permute(1, 0, 2).flatten(start_dim=1).unsqueeze(1)
        mu = self.mu_pred(new_h)
        logvar = self.logvar_pred(new_h)
        return mu, logvar
    
    def decode(self, z : torch.Tensor):
        x = torch.zeros(z.shape[0], 7, self.embedding_dim).to(dtype=z.dtype, device=z.device)
        new_z = z.permute(1, 0, 2)
        out, (h, c) = self.decoder(x, (new_z, new_z))
        return self.ff(out)
    
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean, dtype=mean.dtype)
        return mean + torch.sqrt(logvar.exp()) * eps, eps
    
    def bring_to_latent(self, inputs : torch.Tensor, tokenizer : Tokenizer):
        pass

    def split(self, inputs : torch.Tensor, tokenizer : Tokenizer):
        # Split up the batch into each sequence and each sequence into its operations
        ops = (inputs == tokenizer['rmoveto']) + (inputs == tokenizer['rlineto']) + (inputs == tokenizer['rrcurveto']) + (inputs == tokenizer['endchar'])
        ops_split = torch.split(ops, 1, dim=0)
        op_idx = [torch.argwhere(batch)[:,1] for batch in ops_split]
        op_size = [[(batch_idx[i+1] - batch_idx[i]).item() for i in range(batch_idx.shape[0] - 1)] for batch_idx in op_idx]
        selections = [tablelist[op_idx[i][op]:op_idx[i][op]+op_size[i][op]].to(dtype=torch.int32) for i, tablelist in enumerate(inputs) for op in range(len(op_idx[i])-1)]
        features = torch.nn.utils.rnn.pad_sequence(selections, batch_first=True)
        return features, op_size
    
    def collect_latents(self, batch_size : int, z : torch.Tensor, op_size : list[list[int]], tokenizer : Tokenizer):
        # Reconstruct batch from simplified latent
        selection_lens = np.array([0] + [len(op_size[i]) for i in range(batch_size)]).cumsum()
        latent = [torch.cat([z[selection_lens[op_list_idx] + idx] for idx, _ in enumerate(op_list)] + [torch.zeros((1, self.hidden_dim)).to(device=z.device)], dim=0) for op_list_idx, op_list in enumerate(op_size)]
        padded_latent = torch.nn.utils.rnn.pad_sequence(latent, batch_first=True)
        return padded_latent
    
    def input_to_latent(self, inputs : torch.Tensor, tokenizer : Tokenizer):
        features, op_size = self.split(inputs, tokenizer)
        x = self.embedder(features)
        mu, logvar = self.encode(x)
        z, _ = self.reparameterize(mu, logvar)
        latent = self.collect_latents(inputs.shape[0], z, op_size, tokenizer)
        return latent
    
    def latent_to_input(self, latent : torch.Tensor, tokenizer : Tokenizer):
        pass # TODO: Implement this
    
    def forward(self, inputs : torch.Tensor, tokenizer : Tokenizer):
        features, op_size = self.split(inputs, tokenizer)
        
        # Turn each operation into a single token
        x = self.embedder(features)
        mu, logvar = self.encode(x)
        z, _ = self.reparameterize(mu, logvar)
        features_hat = self.decode(z)

        return features, features_hat, mu, logvar
