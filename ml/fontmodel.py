import torch
import torch.nn as nn


eos_token = -1


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, masked : bool, dropout_rate : float = 0.1) -> nn.Module:
        super(MultiheadAttention, self).__init__()
        
        if embedding_dim % num_heads != 0:
            raise Exception("Embedding dimension must be a multiple of the number of heads.")
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.scale = embedding_dim**(-0.5)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.lifting = nn.Linear(embedding_dim, embedding_dim)
        self.masked = masked
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
        # x is of shape (N, 1 + seq_len, embedding_dim)
        Wq = self.q_proj(q).reshape(q.shape[0], q.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)
        Wk = self.k_proj(k).reshape(k.shape[0], k.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)
        Wv = self.v_proj(v).reshape(v.shape[0], v.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)

        a_t = torch.matmul(Wq.squeeze(0), Wk.squeeze(0).swapaxes(-2, -1)) * self.scale # (N, num_heads, 1 + seq_len, 1 + seq_len)
        a_t = a_t.softmax(dim=-1)
        if self.masked:
            a_t = torch.tril(a_t, diagonal=0) # (N, num_heads, 1 + seq_len, 1 + seq_len)
            a_t = a_t / torch.sum(a_t, dim=-1, keepdim=True)
        attn_vals = torch.matmul(a_t, Wv.squeeze(0)).swapaxes(-3, -2).flatten(-2, -1) # concatenates heads; (N, 1 + seq_len, num_heads * head_size) = (N, 1 + seq_len, embedding_dim)
        mha_out = self.lifting(attn_vals) # (N, 1 + seq_len, embedding_dim)

        return self.dropout(mha_out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.MHA = MultiheadAttention(embedding_dim, num_heads, False, dropout_rate)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mhsa_out = self.MHA(x, x, x)
        x = self.norm_1(mhsa_out + x)
        ff_out = self.ff(x)
        x = self.norm_2(ff_out + x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerDecoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.norm_3 = nn.LayerNorm(embedding_dim)
        
        self.MaskedMHSA = MultiheadAttention(embedding_dim, num_heads, True)
        self.MHA = MultiheadAttention(embedding_dim, num_heads, False)
        
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encodeded source sequence from the encoder
        y (torch.Tensor): the target sequence upon which to generate the next token
        '''
        masked_mhsa_out = self.MaskedMHSA(y, y, y)
        y = self.norm_1(masked_mhsa_out + y)
        mha_out = self.MHA(y, x, x)
        y = self.norm_2(mha_out + y)
        ff_out = self.ff(y)
        y = self.norm_3(ff_out + y)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, embedder : nn.Module, dropout_rate : float, device : torch.device) -> nn.Module:
        super(TransformerEncoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, embedding_dim), requires_grad=True)
        self.token_space = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_encoder_layers = nn.Sequential(
            *[TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        )

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                param.bias.data.fill_(0.01)
        self.transformer_encoder_layers.apply(init_weights)

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        return self.token_space(self.embedder(x)).softmax(dim=-1)

    def forward(self, src : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src (torch.Tensor): the unencoded, unembedded source sequence to encode

        Returns:
        --------
        torch.Tensor: the encoded sequence (batch_size, seq_len + 1, embedding_dim)
        '''
        # x : (batch_size, seq_len, vocab_size)
        embeddings = torch.cat([torch.zeros((src.shape[0], 1, self.embedding_dim)).to(self.device), self.embedder(src)], dim=1)
        embeddings += self.pos_embed[:,:src.shape[1]+1,:]
        return self.transformer_encoder_layers(self.dropout(embeddings))


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, embedder : nn.Module, dropout_rate : float, device : torch.device) -> nn.Module:
        super(TransformerDecoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, embedding_dim), requires_grad=True)
        self.token_space = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embedding_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        )

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                param.bias.data.fill_(0.01)
        self.transformer_decoder_layers.apply(init_weights)

    def forward(self, x : torch.Tensor, tgt : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded sequence from the encoder
        tgt (torch.Tensor): the unencoded, unembedded target sequence to pass directly into the decoder
                          in order to generate the next token

        Returns:
        --------
        torch.Tensor: the probability distribution for next token selection (batch_size, vocab_size)
        '''
        # x : (batch_size, seq_len, vocab_size)
        SOS_token = torch.zeros((x.shape[0], 1, self.embedding_dim)).to(self.device)
        if tgt is not None and tgt.shape[1] != 0:
            embeddings = torch.cat([SOS_token, self.embedder(tgt)], dim=1)
            embeddings += self.pos_embed[:,:tgt.shape[1]+1,:]
        else:
            embeddings = SOS_token
            embeddings += self.pos_embed[:,:1,:]
        embeddings = self.dropout(embeddings)
        for module in self.transformer_decoder_layers:
            embeddings = module(x, embeddings)
        return self.token_space(embeddings[:,-1,:]).softmax(dim=-1)


class FontModel(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float, device : torch.device) -> nn.Module:
        super(FontModel, self).__init__()

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        
        ### If using custom Transformer
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            embedder=self.embedder,
            dropout_rate=dropout_rate,
            device=device
        )
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            embedder=self.embedder,
            dropout_rate=dropout_rate,
            device=device
        )

        ### If using nn.Transformer:
        # self.embedding_dim = embedding_dim
        # self.device = device
        # self.modl = nn.Transformer(
        #     d_model=embedding_dim,
        #     nhead=num_heads,
        #     num_encoder_layers=num_layers,
        #     num_decoder_layers=num_layers,
        #     dim_feedforward=ff_dim,
        #     dropout=dropout_rate,
        #     batch_first=True,
        #     device=device
        # )
        # self.pos_embed = nn.Parameter(torch.zeros(1, 2048, embedding_dim), requires_grad=True)
        # self.tgt_pos_embed = nn.Parameter(torch.zeros(1, 2048, embedding_dim), requires_grad=True)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.tgt_dropout = nn.Dropout(dropout_rate)
        # self.token_space = nn.Linear(embedding_dim, vocab_size)

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        ### If using custom transformer
        return self.encoder.identity_embeddings(x)

        ### If using nn.Transformer
        # return self.token_space(self.dropout(self.embedder(x)))

    # def decode_until_stop(self, src : torch.Tensor, tgt : torch.Tensor = None) -> torch.Tensor:
    #     '''
    #     Parameters:
    #     -----------
    #     src (torch.Tensor): the source sequence to pass to the encoder
    #     tgt (torch.Tensor): the target sequence to pass directly into the decoder
    #                       in order to generate the next token (leave None if generate from start)

    #     Returns:
    #     --------
    #     torch.Tensor: the generated sequence (batch_size, max_seq_len, vocab_size)
    #     '''
    #     # src : (batch_size, seq_len, vocab_size)
    #     encoder_out = self.encoder(src)
    #     decoder_out = self.decoder(encoder_out, tgt)
    #     nxt = torch.multinomial(decoder_out, 1) # Default ancestral
    #     seq = torch.cat([nxt], dim=1)
    #     continue_samples = torch.ones(nxt.shape) * (nxt == eos_token)

    #     while not torch.all(continue_samples == 0) or seq.shape[1] > 9:
    #         decoder_out = self.decoder(encoder_out, seq)
    #         nxt = torch.multinomial(decoder_out, 1) # Default ancestral
    #         seq = torch.cat([seq, nxt], dim=1)
    #         continue_samples = torch.ones(nxt.shape) * (nxt == eos_token)
        
    #     return seq

    def forward(self, src : torch.Tensor, tgt : torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src (torch.Tensor): the source sequence to pass to the encoder
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)

        Returns:
        --------
        torch.Tensor: the probability distribution for next token selection (batch_size, vocab_size)
        '''
        # src : (batch_size, seq_len, vocab_size)

        ### If using custom Transformer
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(encoder_out, tgt)
        return decoder_out

        ### If using nn.Transformer
        # embeddings = torch.cat([torch.zeros((src.shape[0], 1, self.embedding_dim)).to(self.device), self.embedder(src)], dim=1)
        # embeddings += self.pos_embed[:,:src.shape[1]+1,:]
        # embeddings = self.dropout(embeddings)
        # if tgt is None or tgt.shape[1] == 0:
        #     tgt_embeddings = torch.zeros((src.shape[0], 1, self.embedding_dim)).to(self.device)
        #     tgt_embeddings += self.tgt_pos_embed[:,:1,:]
        # else:
        #     tgt_embeddings = torch.cat([torch.zeros((tgt.shape[0], 1, self.embedding_dim)).to(self.device), self.embedder(tgt)], dim=1)
        #     tgt_embeddings += self.tgt_pos_embed[:,:tgt.shape[1]+1,:]
        # tgt_embeddings = self.tgt_dropout(tgt_embeddings)
        # return self.token_space(self.modl(embeddings, tgt_embeddings)[:,-1,:]).softmax(dim=-1)