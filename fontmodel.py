import torch
import torch.nn as nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, masked : bool):
        super(MultiheadSelfAttention, self).__init__()
        
        if embedding_dim % num_heads != 0:
            raise Exception("Embedding dimension must be a multiple of the number of heads.")
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.scale = embedding_dim**(-0.5)

        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)
        self.lifting = nn.Linear(embedding_dim, embedding_dim)
        self.masked = masked
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x is of shape (N, 1 + seq_len, embedding_dim)
        qkv = self.qkv_proj(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4) # (3, N, num_heads, 1 + seq_len, head_size)
        # ^ essentially, each head gets its own query, key, and value projection (of the same size as the head) of each word in the sequence for each sequence in the batch
        q, k, v = torch.chunk(qkv, 3) # now sets of (N, num_heads, 1 + seq_len, head_size); well, with an extra dimension of 1 at index 0

        a_t = torch.matmul(q.squeeze(0), k.squeeze(0).swapaxes(-2, -1)) * self.scale # (N, num_heads, 1 + seq_len, 1 + seq_len)
        a_t = a_t.softmax(dim=-1)
        if self.masked:
            a_t = torch.tril(a_t) # (N, num_heads, 1 + seq_len, 1 + seq_len)
            a_t = a_t / torch.sum(a_t, dim=-1, keepdim=True)
        attn_vals = torch.matmul(a_t, v.squeeze(0)).swapaxes(-3, -2).flatten(-2, -1) # concatenates heads; (N, 1 + seq_len, num_heads * head_size) = (N, 1 + seq_len, embedding_dim)
        mha_out = self.lifting(attn_vals) # (N, 1 + seq_len, embedding_dim)

        return self.dropout(mha_out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.MHSA = MultiheadSelfAttention(embedding_dim, num_heads, False)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        mhsa_out = self.MHSA(x)
        x = self.norm_1(mhsa_out + x)
        ff_out = self.ff(x)
        x = self.norm_2(ff_out + x)
        return x


class FontModel(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int, ff_dim : int, device : torch.device) -> nn.Module:
        super(FontModel, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # self.embedder = nn.Linear(vocab_size, embedding_dim)
        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.reverse_embedder = nn.Linear(embedding_dim, vocab_size) # for pre-training

        self.pos_embed = nn.Parameter(torch.zeros(1, 2048, embedding_dim), requires_grad=True)

        self.transformer_encoder_layers = nn.Sequential(
            *[TransformerEncoderLayer(embedding_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform(param.weight)
                param.bias.data.fill_(0.01)
        self.transformer_encoder_layers.apply(init_weights)

        self.token_space = nn.Linear(embedding_dim, vocab_size)

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        return self.reverse_embedder(self.embedder(x)).softmax(dim=-1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x : (batch_size, seq_len, vocab_size)
        embeddings = torch.cat([torch.zeros((x.shape[0], 1, self.embedding_dim)).to(self.device), self.embedder(x)], dim=1)
        embeddings += self.pos_embed[:,:x.shape[1]+1,:]
        encoder_out = self.transformer_encoder_layers(embeddings)
        
        tokenized = self.token_space(encoder_out).softmax(dim=-1) # (batch_size, seq_len, vocab_size)
        return tokenized[:,0]