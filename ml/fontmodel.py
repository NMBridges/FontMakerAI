import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from enum import Enum
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class DecodeType(Enum):
    ANCESTRAL = 0 # ancestral
    BEAM = 1 # beam search
    # VITERBI=2


class SamplingType(Enum):
    MULTINOMIAL = 0 # default softmax
    TEMPERATURE = 1 # temperature-based softmax
    GREEDY = 2 # greedy sampling
    TOPK = 3 # top-k sampling
    TOPP = 4 # top-p (nucleus) sampling

    
class TransformerScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, dim_embed: int, warmup_steps: int,
                 last_epoch: int = -1, verbose: bool = False):
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        
        super(TransformerScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.dim_embed**(-0.5) * min(self._step_count**(-0.5), self._step_count * self.warmup_steps**(-1.5))
        return [lr] * self.num_param_groups


class DecodeInstruction:
    def __init__(self, decode_type : DecodeType, sampling_type : SamplingType, **kwargs):
        self.decode_type = decode_type
        self.sampling_type = sampling_type

        self.max_seq_len = kwargs['max_seq_len']
        if decode_type == DecodeType.ANCESTRAL:
            pass
        elif decode_type == DecodeType.BEAM:
            self.beam_size = kwargs['beam_size']

        if sampling_type == SamplingType.MULTINOMIAL:
            pass
        elif sampling_type == SamplingType.TEMPERATURE:
            self.temp = kwargs['temp']
        elif sampling_type == SamplingType.GREEDY:
            pass
        elif sampling_type == SamplingType.TOPK:
            self.k = kwargs['k']
        elif sampling_type == SamplingType.TOPP:
            self.p = kwargs['p']


class LearnedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim : int, max_seq_len : int):
        super(LearnedAbsolutePositionalEmbedding, self).__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim), requires_grad=True)

    def forward(self, x):
        return x + self.pos_embed[:,:x.shape[1],:]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim : int, max_seq_len : int):
        super(RotaryPositionalEmbedding, self).__init__()

        coeffs = 1 / torch.pow(10000, torch.arange(0, embedding_dim, 2).float() / embedding_dim)
        angles = torch.einsum('i,j->ij', torch.arange(max_seq_len), coeffs)
        self.pos_embed = nn.Parameter(torch.stack((angles.cos(), angles.sin()), dim=2), requires_grad=False) # (max_seq_len, embed_dim, 2)

    def forward(self, x):
        '''x : (bs, num_heads, seq_len, head_size)'''
        cos = self.pos_embed[None,:x.shape[2],:,0].view((1,x.shape[2], x.shape[1], x.shape[3] // 2)).permute(0, 2, 1, 3) # (1, num_heads, seq_len, head_size/2)
        sin = self.pos_embed[None,:x.shape[2],:,1].view((1,x.shape[2], x.shape[1], x.shape[3] // 2)).permute(0, 2, 1, 3) # (1, num_heads, seq_len, head_size/2)

        x_1 = x[...,0::2] * cos - x[...,1::2] * sin
        x_2 = x[...,1::2] * cos + x[...,0::2] * sin
        return torch.stack([x_1, x_2], dim=4).flatten(start_dim=3)
    

class RotaryAttention(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int = 8, dropout_rate : float = 0.0, max_seq_len : int = 2000):
        super(RotaryAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.dropout = dropout_rate
        self.q_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_size, embedding_dim, bias=False)
        self.scale = self.head_size**-0.5
        self.pos_embed = RotaryPositionalEmbedding(embedding_dim, max_seq_len)

    def forward(self, x : torch.Tensor):
        q = self.q_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        q = self.pos_embed(q)
        k = self.pos_embed(k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        out_vals = self.out_proj(out.permute(0, 2, 1, 3).flatten(start_dim=-2)) # (bs, seq_len, d)
        return out_vals
    

class MHSDPA(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int = 8, dropout_rate : float = 0.1):
        super(MHSDPA, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads
        self.dropout = dropout_rate
        self.q_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(embedding_dim, num_heads * self.head_size, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_size, embedding_dim, bias=False)
        self.scale = self.head_size**-0.5

    def forward(self, x : torch.Tensor):
        q = self.q_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        k = self.k_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        v = self.v_proj(x).unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3) # (bs, seq_len, d) -> (bs, num_heads, seq_len, head_size)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        out_vals = self.out_proj(out.permute(0, 2, 1, 3).flatten(start_dim=-2)) # (bs, seq_len, d)
        return out_vals
    

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.RMSNorm(embedding_dim)
        self.norm_2 = nn.RMSNorm(embedding_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.MHA = MHSDPA(embedding_dim, num_heads, dropout_rate)
        # self.MaskedMHSA = RotaryAttention(embedding_dim, num_heads, dropout_rate)
        self.ff = SwiGLU_FNN(embedding_dim, ff_dim)

    def forward(self, x : torch.Tensor, src_mask : torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the partially encoded source sequence from the previous layer
        src_mask (torch.Tensor): the mask for the source sequence
        '''
        norm_x = self.norm_1(x)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = self.dropout_1(self.MHA(norm_x)) + x
        x = self.dropout_2(self.ff(self.norm_2(x))) + x
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1, max_seq_len : int = 2000) -> nn.Module:
        super(TransformerDecoderLayer, self).__init__()

        self.norm_1 = nn.RMSNorm(embedding_dim)
        self.norm_2 = nn.RMSNorm(embedding_dim)
        self.norm_3 = nn.RMSNorm(embedding_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)
        # self.MaskedMHSA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.MaskedMHSA = RotaryAttention(embedding_dim, num_heads, dropout_rate, max_seq_len)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ff = SwiGLU_FNN(embedding_dim, ff_dim)


    def forward(self, x : torch.Tensor, y : torch.Tensor, src_mask : torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded source sequence from the encoder
        y (torch.Tensor): the target sequence upon which to generate the next token
        '''
        # Masked MHSA
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            # masked_mhsa_out = self.MaskedMHSA(norm_y, norm_y, norm_y, attn_mask=tgt_mask, need_weights=False)[0]
            # masked_mhsa_out = self.MaskedMHSA(q, k, norm_y, attn_mask=tgt_mask, is_causal=is_causal, need_weights=False)[0]
            y = self.dropout_1(self.MaskedMHSA(self.norm_1(y))) + y
        # MHA
        if x is not None and x.shape[1] != 0:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = self.dropout_2(self.MHA(self.norm_2(y), x, x, attn_mask=src_mask, need_weights=False)[0]) + y
        # Feedforward
        y = self.dropout_3(self.ff(self.norm_3(y))) + y
        return y
    

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

    def forward(self, x):
        x = self.res_double_conv(x)
        x = self.pool(x)
        return x
    

class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResUpBlock, self).__init__()
        self.unpool = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.res_double_conv = ResDoubleConv(in_channels, out_channels, kernel_size, 1, (kernel_size - 1) // 2, dilation=dilation)

    def forward(self, x):
        x = self.unpool(x)
        x = self.res_double_conv(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, dropout_rate : float, device : torch.device) -> nn.Module:
        super(TransformerEncoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.zero_tensor = nn.Parameter(torch.ones((10000, 1, embedding_dim)), requires_grad=False)

        # self.embedder = embedder
        self.embedder = nn.Sequential(
            # nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid(),
            nn.Conv2d(1, embedding_dim, kernel_size=(8, 8), stride=(8, 8)),
        )
        # self.embedder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # 128x128
        #     nn.RMSNorm((16, 128, 128)),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1), # 64x64
        #     nn.RMSNorm((64, 64, 64)),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1), # 32x32
        #     nn.RMSNorm((256, 32, 32)),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(256, embedding_dim, kernel_size=3, stride=2, padding=1) # 16x16
        # )
        # self.embedder = nn.Sequential(
        #     ResDownBlock(1, 64, kernel_size=3, dilation=1), # 64x64
        #     nn.RMSNorm((64, 64, 64)),
        #     nn.LeakyReLU(0.2),
        #     ResDownBlock(64, 256, kernel_size=3, dilation=1), # 32x32
        #     nn.RMSNorm((256, 32, 32)),
        #     nn.LeakyReLU(0.2),
        #     ResDownBlock(256, embedding_dim, kernel_size=3, dilation=1) # 16x16
        # )
        self.pretrain_reverse_ae = nn.Sequential(
            ResUpBlock(embedding_dim, 512, kernel_size=3, dilation=1), # 2x2
            nn.RMSNorm((512, 2, 2)),
            nn.LeakyReLU(0.2),
            ResUpBlock(512, 256, kernel_size=3, dilation=1), # 4x4
            nn.RMSNorm((256, 4, 4)),
            nn.LeakyReLU(0.2),
            ResUpBlock(256, 128, kernel_size=3, dilation=1), # 8x8
            nn.RMSNorm((128, 8, 8)),
            nn.LeakyReLU(0.2),
            ResUpBlock(128, 128, kernel_size=5, dilation=1), # 16x16
            nn.RMSNorm((128, 16, 16)),
            nn.LeakyReLU(0.2),
            ResUpBlock(128, 64, kernel_size=5, dilation=1), # 32x32
            nn.RMSNorm((64, 32, 32)),
            nn.LeakyReLU(0.2),
            ResUpBlock(64, 32, kernel_size=5, dilation=1), # 64x64
            nn.RMSNorm((32, 64, 64)),
            nn.LeakyReLU(0.2),
            ResUpBlock(32, 16, kernel_size=3, dilation=1), # 128x128
            nn.RMSNorm((16, 128, 128)),
            nn.LeakyReLU(0.2),
            ResDoubleConv(16, 1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Tanh()
        )

        # Learned position embeddings
        self.pos_embed = LearnedAbsolutePositionalEmbedding(embedding_dim, 16*16+1)
        # self.pos_embed = RotaryPositionalEmbedding(embedding_dim, 16*16)
        
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_encoder_layers = nn.Sequential(
            *[TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_layers)]
        )
        self.norm_final = nn.RMSNorm(embedding_dim)

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                if param.bias is not None:
                    param.bias.data.fill_(0.00)
        self.transformer_encoder_layers.apply(init_weights)

    def forward(self, src : torch.Tensor, src_mask : torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src (torch.Tensor): the unencoded, unembedded source sequence to encode

        Returns:
        --------
        torch.Tensor: the encoded sequence (batch_size, 16*16, embedding_dim)
        '''
        # x : (batch_size, 1, 64, 64)
        embeddings = self.embedder(src).permute(0, 2, 3, 1).flatten(start_dim=-3, end_dim=-2)
        embeddings = self.pos_embed(embeddings)
        embeddings = self.dropout(embeddings)
        # return embeddings
        return self.norm_final(self.transformer_encoder_layers(embeddings))

    def pretrain(self, src : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src (torch.Tensor): the unencoded, unembedded source sequence to encode

        Returns:
        --------
        torch.Tensor: the encoded sequence (batch_size, 16*16 + 1, embedding_dim)
        '''
        # x : (batch_size, 1, 64, 64)
        embeddings = self.embedder(src).permute(0, 2, 3, 1).flatten(start_dim=-3, end_dim=-2)
        embeddings = torch.cat([embeddings, self.zero_tensor[:src.shape[0]]], dim=1)
        embeddings = self.pos_embed(embeddings)
        out = self.norm_final(self.transformer_encoder_layers(self.dropout(embeddings)))[:,-1,:]
        return self.pretrain_reverse_ae(out.view(out.shape[0], self.embedding_dim, 1, 1))


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers : int, vocab_size : int, embedding_dim : int, num_heads : int,
                    ff_dim : int, max_seq_len : int, sos_token : int = 1, eos_token : int = 2,
                        dropout_rate : float = 0.1, device : torch.device = torch.device('cpu')) -> nn.Module:
        super(TransformerDecoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        max_batch_size = 512
        self.sos_token = nn.Parameter(torch.Tensor([[sos_token]]).repeat((max_batch_size, 1)).int(), requires_grad=False)
        self.eos_token = nn.Parameter(torch.Tensor([[eos_token]]).repeat((max_batch_size, 1)).int(), requires_grad=False)
        self.pad_token = nn.Parameter(torch.Tensor([[0]]).repeat((max_batch_size, 1)).int(), requires_grad=False)

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.inverse_embedder = nn.Linear(embedding_dim, vocab_size, bias=False)

        self.command_encoder = nn.Linear(embedding_dim * 7, embedding_dim, bias=False)
        self.command_decoder = nn.Linear(embedding_dim, 7 * embedding_dim, bias=False)

        self.command_decoder_2a = nn.Linear(embedding_dim, 32, bias=False)
        self.command_decoder_2b = nn.Sequential(
            # nn.Linear(embedding_dim, 6, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, 1, bias=False),
        )

        # Learned position embeddings
        # self.pos_embed = LearnedAbsolutePositionalEmbedding(embedding_dim, max_seq_len // 7)
        # self.pos_embed = RotaryPositionalEmbedding(embedding_dim, 10000)

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(embedding_dim, num_heads, ff_dim, dropout_rate, max_seq_len // 7) for _ in range(num_layers)]
        )
        self.norm_final = nn.RMSNorm(embedding_dim)

        # Source: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch solution
        def init_weights(param):
            if isinstance(param, nn.Linear):
                torch.nn.init.xavier_uniform_(param.weight)
                if param.bias is not None:
                    param.bias.data.fill_(0.00)
        self.transformer_decoder_layers.apply(init_weights)

    def forward(self, x : torch.Tensor, tgt : torch.Tensor, src_mask : torch.Tensor = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded sequence from the encoder
        tgt (torch.Tensor): the unencoded, unembedded target sequence to pass directly into the decoder
                          in order to generate the next token
        is_causal (bool): whether or not to use a causal mask
        Returns:
        --------
        torch.Tensor: the logits for token selection (batch_size, seq_len + 1, vocab_size)
        '''
        # x : (batch_size, seq_len, vocab_size)
        embeddings = self.embedding_space(tgt)
        sos_embedded = self.embed(self.sos_token[:tgt.shape[0]])
        if tgt.shape[1] != 0:
            embeddings = torch.cat([sos_embedded, embeddings], dim=1)
        elif tgt.shape[1] == 0:
            embeddings = sos_embedded
        # embeddings = self.pos_embed(embeddings)
        embeddings = self.dropout(embeddings)
        for module in self.transformer_decoder_layers:
            embeddings = module(x, embeddings, src_mask)
        return self.token_space(self.norm_final(embeddings))
    
    def embed(self, x : torch.Tensor) -> torch.Tensor:
        return self.embedder(x)# * (self.embedding_dim ** 0.5)
    
    def embedding_space(self, x : torch.Tensor) -> torch.Tensor:
        x = self.embed(x) # (batch_size, seq_len, embedding_dim)
        x = self.command_encoder(x.unflatten(dim=-2, sizes=(-1, 7)).flatten(start_dim=-2)) # (batch_size, seq_len // 7, embedding_dim)
        return x
    
    def token_space(self, x : torch.Tensor) -> torch.Tensor:
        # x : (batch_size, seq_len // 7, embedding_dim)
        x = self.command_decoder(x) # (batch_size, seq_len // 7, 7 * embedding_dim)
        # x = x.unflatten(dim=-1, sizes=(7, -1)).flatten(start_dim=-3, end_dim=-2)
        x = x.view((x.shape[0], -1, self.embedding_dim)) # (batch_size, seq_len, embedding_dim)
        x = self.inverse_embedder(x) # (batch_size, seq_len, vocab_size)
        return x
    
    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        return self.token_space(self.dropout(self.norm_final(self.embedding_space(x))))
    
    def _step(self, x : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None,
                    scores : torch.Tensor = None, continue_samples : torch.Tensor = None,
                        src_mask : torch.Tensor = None) -> torch.Tensor:
        '''
        Decodes a single step of the sequence.

        Parameters:
        -----------
        x (torch.Tensor): the encoded source sequence from the encoder
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)
        instruction (DecodeInstruction): the data structure containing instructions for how to decode
        scores (torch.Tensor): the running scores of each of the hypotheses
        continue_samples (torch.Tensor): whether or not to continue sampling this batch item (0 or 1)
        src_mask (torch.Tensor): the mask for the source sequence

        Returns:
        --------
        torch.Tensor: the generated sequence (batch_size, seq_len, ?num_hypotheses_per_batch_item?)
        (in-place modification of scores) torch.Tensor: the new scores of the hypotheses 
                                                    (batch_size, ?num_hypotheses_per_batch_item?)
        '''
        ### ANCESTRAL
        if instruction.decode_type == DecodeType.ANCESTRAL:
            decoder_out = self.forward(x, tgt, src_mask)

            select_last = 7

            if instruction.sampling_type == SamplingType.MULTINOMIAL:
                nxt = torch.multinomial(
                    input=decoder_out[:,-select_last:,:].softmax(dim=-1),
                    num_samples=1,
                    replacement=True
                ) # Default ancestral
            
            elif instruction.sampling_type == SamplingType.TEMPERATURE:
                nxt = torch.multinomial(
                    input=(decoder_out[:,-select_last:,:] / instruction.temp).softmax(dim=-1),
                    num_samples=1,
                    replacement=True
                )
            
            elif instruction.sampling_type == SamplingType.GREEDY:
                nxt = torch.argmax(decoder_out[:,-select_last:,:], dim=-1, keepdim=True)
                
            elif instruction.sampling_type == SamplingType.TOPK:
                k = instruction.k
                top_k = torch.topk(decoder_out[:,-select_last:,:], k) # ([values], [indices])
                probs = top_k[0].softmax(dim=-1)
                indices_of_k = torch.multinomial(
                    input=probs,
                    num_samples=1,
                    replacement=True
                )
                nxt = torch.gather(top_k[1], dim=-1, index=indices_of_k)

            elif instruction.sampling_type == SamplingType.TOPP:
                probs = decoder_out[:,-select_last:,:].softmax(dim=-1)
                indices = torch.argsort(probs, dim=-1, descending=False)
                cum_sum = torch.cumsum(torch.gather(probs, -1, indices), dim=-1)
                
                p = instruction.p
                if p > 0:
                    mask = (cum_sum >= (1 - p))
                    norm_probs = torch.gather(probs, 1, indices)
                    norm_probs /= torch.sum(norm_probs * mask, dim=-1, keepdim=True)
                    indices_of_p = torch.multinomial(
                        input=norm_probs * mask,
                        num_samples=1,
                        replacement=True
                    )
                    nxt = torch.gather(indices, dim=-1, index=indices_of_p)
                else:
                    nxt = torch.argmax(probs, dim=-1, keepdim=True)
            
            else:
                raise Exception(f"Invalid sampling type {instruction.sampling_type}")
            
            # Mask out bad arguments
            if nxt[0,0,0] == 4 or nxt[0,0,0] == 7:
                nxt[0,1:5,:] = self.pad_token[:4]
            elif nxt[0,0,0] == 31:
                nxt[0,1:,:] = self.pad_token[:6]

            if tgt is None:
                seq = torch.cat([nxt], dim=1)#.to(torch.int16)
            else:
                seq = torch.cat([tgt, nxt.flatten(start_dim=-2)], dim=1)#.to(torch.int16)
            
            return seq

        ### BEAM SEARCH
        elif instruction.decode_type == DecodeType.BEAM:
            if tgt is None:
                (B, length, hypo) = (x.shape[0], 1, 1)
                decoder_out = self.forward(
                    x, # (batch_size, src_seq_len, embed_dim)
                    tgt, # None
                    src_mask,
                    None,
                    is_causal=False
                )
            else:
                (B, length, hypo) = tgt.shape
                decoder_out = self.forward(
                    x.repeat((instruction.beam_size, 1, 1)),
                    tgt.permute(0, 2, 1).flatten(end_dim=1),
                    src_mask,
                    None,
                    is_causal=False
                )
            # decoder_out : (batch_size * hypotheses, seq_len, logits)
            if instruction.sampling_type == SamplingType.MULTINOMIAL:
                probs = decoder_out[:,-1,:].softmax(dim=-1)
                log_p_nxt = torch.log(probs)
                nxt = torch.multinomial(
                    input=probs,
                    num_samples=instruction.beam_size,
                    replacement=False
                ) # Default ancestral
            
            elif instruction.sampling_type == SamplingType.TEMPERATURE:
                probs = (decoder_out[:,-1,:] / instruction.temp).softmax(dim=-1)
                log_p_nxt = torch.log(probs)
                nxt = torch.multinomial(
                    input=probs,
                    num_samples=instruction.beam_size,
                    replacement=False
                ) # (batch_size * hypotheses, beam_size)
            
            elif instruction.sampling_type == SamplingType.GREEDY:
                log_p_nxt = decoder_out[:,-1,0] * 0
                nxt = torch.argmax(decoder_out[:,-1,:], dim=-1, keepdim=True) # Pointless
                log_p_nxt.scatter_(dim=-1, index=nxt, scr=torch.zeros_like(nxt))
            
            elif instruction.sampling_type == SamplingType.TOPK:
                k = instruction.k
                top_k = torch.topk(decoder_out[:,-1,:], k) # ([values], [indices])
                probs = top_k[0].softmax(dim=-1)
                indices_of_k = torch.multinomial(
                    input=probs,
                    num_samples=instruction.beam_size,
                    replacement=True
                )
                nxt = torch.gather(top_k[1], dim=-1, index=indices_of_k)
                all_probs = torch.zeros_like(decoder_out[:,-1,:])
                all_probs.scatter_(dim=-1, index=top_k[1], src=probs)
                log_p_nxt = torch.log(all_probs)

            elif instruction.sampling_type == SamplingType.TOPP:
                probs = decoder_out[:,-1,:].softmax(dim=-1)
                indices = torch.argsort(probs, dim=-1, descending=False)
                cum_sum = torch.cumsum(torch.gather(probs, -1, indices), dim=-1)
                
                p = instruction.p
                if p > 0:
                    mask = (cum_sum >= (1 - p))
                    norm_probs = torch.gather(probs, 1, indices)
                    norm_probs /= torch.sum(norm_probs * mask, dim=-1, keepdim=True)
                    indices_of_p = torch.multinomial(
                        input=norm_probs * mask,
                        num_samples=instruction.beam_size,
                        replacement=True
                    )
                    
                    all_probs = torch.zeros_like(probs)
                    all_probs.scatter_(dim=-1, index=indices, src=norm_probs)
                    log_p_nxt = torch.log(all_probs)
                    nxt = torch.gather(indices, dim=-1, index=indices_of_p)
                else:
                    nxt = torch.argmax(probs, dim=-1, keepdim=True)
                    log_p_nxt = torch.log(nxt * 0)
                    log_p_nxt.scatter_(dim=-1, index=nxt, src=torch.zeros_like(nxt))
            
            else:
                raise Exception(f"Invalid sampling type {instruction.sampling_type}")

            # Expand hypotheses
            nxt_log_prob = torch.gather(log_p_nxt, dim=-1, index=nxt).unflatten(dim=0, sizes=(B, hypo)) # (batch_size, num_hypo, beam_size)
            new_scores = (scores.unsqueeze(dim=-2) + nxt_log_prob * continue_samples[:,None,None]).flatten(start_dim=1) # (batch_size, num_hypo * beam_size)
            # Trim
            top_k = torch.topk(new_scores, instruction.beam_size, dim=-1)[1] # (batch_size, beam_size)
            nxt = torch.gather(
                input=nxt.unflatten(dim=0, sizes=(B, hypo)).flatten(start_dim=1), # (batch_size, num_hypo * beam_size)
                dim=-1,
                index=top_k # (batch_size, beam_size)
            )[:,None,:] # (batch_size, 1, beam_size)

            torch.gather(
                input=new_scores,
                dim=-1,
                index=top_k, # (batch_size, beam_size)
                out=scores
            ) # (batch_size, beam_size)
            
            if tgt is None:
                seq = torch.cat([nxt], dim=1)
            else:
                indices = (top_k // instruction.beam_size)[:,None,:]
                bef = torch.gather(input=tgt, dim=-1, index=indices.expand_as(tgt))
                seq = torch.cat([bef, nxt], dim=1)
            
            return seq
        else:
            raise Exception(f"Invalid decode type {instruction.decode_type}")
    
    def decode(self, x : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None) -> torch.Tensor:
        '''
        Decodes a sequence until the EOS (end of sequence) token is reached or the max sequence length is reached.
        
        Parameters:
        -----------
        x (torch.Tensor): the encoded source sequence from the encoder (batch_size, src_len, embedding_dim)
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)
        instruction (DecodeInstruction): the data structure containing instructions for how to decode

        Returns:
        --------
        torch.Tensor: the generated sequence (batch_size, max_seq_len)
        '''
        attempts = 0
        while attempts < 10:
            if instruction.decode_type == DecodeType.BEAM:
                scores = torch.zeros((x.shape[0], instruction.beam_size)).to(x.device)
            else:
                scores = torch.zeros((x.shape[0],)).to(x.device)
            
            src_mask = None#torch.zeros((x.shape[0], 1, 1, x.shape[1])).to(x.device)
            continue_samples = torch.ones(x.shape[0],).to(x.device)
            seq = self._step(x, tgt, instruction, scores, continue_samples, src_mask)

            if instruction.decode_type == DecodeType.BEAM:
                continue_samples = continue_samples * torch.all(seq[:,-1] != self.eos_token[:x.shape[0]], dim=1)
            else:
                continue_samples = continue_samples * (seq[:,-1] != self.eos_token[:x.shape[0]])
            
            if torch.any(continue_samples == 0):
                attempts += 1
            else:
                break
        if attempts == 10:
            raise Exception("Decoding kept generating EOS token at start.")
        
        while torch.any(continue_samples == 1) and seq.shape[1] < instruction.max_seq_len:
            src_mask = None#torch.zeros((x.shape[0], 1, 1, x.shape[1])).to(x.device)
            seq = self._step(x, seq, instruction, scores, continue_samples, src_mask)

            if instruction.decode_type == DecodeType.BEAM:
                continue_samples = continue_samples * torch.all(seq[:,-1] != self.eos_token[:x.shape[0]], dim=1)
            else:
                continue_samples = continue_samples * (seq[:,-7:] != self.eos_token[:x.shape[0]]).any(dim=-1)

        if instruction.decode_type == DecodeType.BEAM:
            # Best sequence
            outs = []
            for b_idx in range(x.shape[0]):
                max_idx = -1
                max_score = -10000000000
                for hypo in range(instruction.beam_size):
                    if self.eos_token[0] in seq[b_idx,:,hypo] and scores[b_idx,hypo] > max_score:
                        max_idx = hypo
                        max_score = scores[b_idx,hypo] > max_score
                if max_idx == -1:
                    print(f"All decoded hypotheses do not have an EOS token for batch index {b_idx}. Passing.")
                    continue
                index_of_eos = (seq[b_idx,:,max_idx] == self.eos_token[0]).nonzero(as_tuple=True)[0][0]
                outs.append(seq[b_idx,:index_of_eos+1,max_idx])
        else:
            outs = []
            for b_idx in range(x.shape[0]):
                if self.eos_token[0] not in seq[b_idx,:]:
                    print(f"Decoded sequence does not have an EOS token for batch index {b_idx}. Passing.")
                    # continue
                    index_of_eos = seq.shape[1]
                else:
                    index_of_eos = (seq[b_idx,:] == self.eos_token[0]).nonzero(as_tuple=True)[0][0]
                outs.append(seq[b_idx,:index_of_eos+1])
        
        return outs


class FontModel(nn.Module):
    def __init__(self, num_enc_layers : int, num_dec_layers : int, vocab_size : int, embedding_dim : int,
                 num_heads : int, ff_dim : int, dropout_rate : float, max_seq_len : int, device : torch.device) -> nn.Module:
        super(FontModel, self).__init__()
        
        self.encoder = TransformerEncoder(
            num_layers=num_enc_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            device=device
        )
        self.decoder = TransformerDecoder(
            num_layers=num_dec_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
            device=device
        )

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        ### If using custom transformer
        return self.decoder.identity_embeddings(x)

    def decode(self, src : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        src (torch.Tensor): the source sequence to pass to the encoder
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)
        instruction (DecodeInstruction): the instruction for how to decode

        Returns:
        --------
        torch.Tensor: the generated sequence (batch_size, max_seq_len, vocab_size)
        '''
        # src : (batch_size, in_seq_len, vocab_size) | None
        # tgt : (batch_size, out_seq_len) | None
        if tgt is None:
            if src is None:
                # Don't know batch size; assume 1
                tgt = torch.zeros((1, 0), dtype=torch.int32).to(x.device)
            else:
                tgt = torch.zeros((src.shape[0], 0), dtype=torch.int32).to(src.device)
        x = self.encoder(src)
        return self.decoder.decode(x, tgt, instruction)
        

    def forward(self, src : torch.Tensor, tgt : torch.Tensor = None) -> torch.Tensor:
        '''
        Note: this uses causal attention masks for the decoder.

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
        x = self.encoder(src)
        if tgt is None:
            if src is None:
                tgt = torch.zeros((1, 0), dtype=torch.int32).to(src.device)
            else:
                tgt = torch.zeros((src.shape[0], 0), dtype=torch.int32).to(src.device)
        # x = torch.zeros((src.shape[0], 0, self.embedder.embedding_dim)).to(src.device)

        # src_mask = (src != 0).to(src.device).unsqueeze(1).unsqueeze(2)
        src_mask = None
        
        decoder_out = self.decoder(x, tgt, src_mask)
        return decoder_out