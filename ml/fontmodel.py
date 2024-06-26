import torch
import torch.nn as nn
from enum import Enum
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

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


# class MultiheadAttention(nn.Module):
#     def __init__(self, embedding_dim : int, num_heads : int, masked : bool, dropout_rate : float = 0.1) -> nn.Module:
#         super(MultiheadAttention, self).__init__()
        
#         if embedding_dim % num_heads != 0:
#             raise Exception("Embedding dimension must be a multiple of the number of heads.")
#         self.num_heads = num_heads
#         self.head_size = embedding_dim // num_heads
#         self.scale = embedding_dim**(-0.5)

#         self.q_proj = nn.Linear(embedding_dim, embedding_dim)
#         self.k_proj = nn.Linear(embedding_dim, embedding_dim)
#         self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
#         self.lifting = nn.Linear(embedding_dim, embedding_dim)
#         self.masked = masked
#         self.softmax = nn.Softmax()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
#         # x is of shape (N, 1 + seq_len, embedding_dim)
#         Wq = self.q_proj(q).reshape(q.shape[0], q.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)
#         Wk = self.k_proj(k).reshape(k.shape[0], k.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)
#         Wv = self.v_proj(v).reshape(v.shape[0], v.shape[1], self.num_heads, self.head_size).permute(0, 2, 1, 3)

#         a_t = torch.matmul(Wq.squeeze(0), Wk.squeeze(0).swapaxes(-2, -1)) * self.scale # (N, num_heads, 1 + seq_len, 1 + seq_len)
#         a_t = a_t.softmax(dim=-1)
#         if self.masked:
#             a_t = torch.tril(a_t, diagonal=0) # (N, num_heads, 1 + seq_len, 1 + seq_len)
#             a_t = a_t / torch.sum(a_t, dim=-1, keepdim=True)
#         attn_vals = torch.matmul(a_t, Wv.squeeze(0)).swapaxes(-3, -2).flatten(-2, -1) # concatenates heads; (N, 1 + seq_len, num_heads * head_size) = (N, 1 + seq_len, embedding_dim)
#         mha_out = self.lifting(attn_vals) # (N, 1 + seq_len, embedding_dim)

#         return self.dropout(mha_out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate)
        # self.MHA = MultiheadAttention(embedding_dim, num_heads, False, dropout_rate)
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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerDecoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.norm_3 = nn.LayerNorm(embedding_dim)
        
        # self.MaskedMHSA = MultiheadAttention(embedding_dim, num_heads, True)
        # self.MHA = MultiheadAttention(embedding_dim, num_heads, False)
        
        self.MaskedMHSA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        
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
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(y.shape[1], y.device)
        masked_mhsa_out, _ = self.MaskedMHSA(y, y, y, attn_mask=causal_mask, is_causal=True, need_weights=False)
        y = self.norm_1(masked_mhsa_out + y)
        mha_out, _ = self.MHA(y, x, x, need_weights=False)
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
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True)
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
                    ff_dim : int, embedder : nn.Module = None, sos_token : int = 1, eos_token : int = 2,
                        dropout_rate : float = 0.1, device : torch.device = torch.device('cpu')) -> nn.Module:
        super(TransformerDecoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sos_token = torch.Tensor([[sos_token]]).repeat((256, 1)).int().to(device)
        self.eos_token = torch.Tensor([[eos_token]]).repeat((256, 1)).int().to(device)

        self.embedder = embedder
        if embedder is None:
            self.embedder = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True)
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

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        return self.token_space(self.embedder(x))

    @torch.no_grad()
    def _step(self, x : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None,
                    scores : torch.Tensor = None) -> torch.Tensor:
        '''
        Decodes a single step of the sequence.

        Parameters:
        -----------
        x (torch.Tensor): the encoded source sequence from the encoder
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)
        instruction (DecodeInstruction): the data structure containing instructions for how to decode
        scores (torch.Tensor): the running scores of each of the hypotheses

        Returns:
        --------
        torch.Tensor: the generated sequence (batch_size, seq_len, ?num_hypotheses_per_batch_item?)
        (in-place modification of scores) torch.Tensor: the new scores of the hypotheses 
                                                    (batch_size, ?num_hypotheses_per_batch_item?)
        '''
        ### ANCESTRAL
        if instruction.decode_type == DecodeType.ANCESTRAL:
            decoder_out = self.forward(x, tgt)

            if instruction.sampling_type == SamplingType.MULTINOMIAL:
                nxt = torch.multinomial(
                    input=decoder_out[:,-1,:].softmax(dim=-1),
                    num_samples=1,
                    replacement=True
                ) # Default ancestral
            
            elif instruction.sampling_type == SamplingType.TEMPERATURE:
                nxt = torch.multinomial(
                    input=(decoder_out[:,-1,:] / instruction.temp).softmax(dim=-1),
                    num_samples=1,
                    replacement=True
                )
            
            elif instruction.sampling_type == SamplingType.GREEDY:
                nxt = torch.argmax(decoder_out[:,-1,:], dim=-1, keepdim=True)
            
            elif instruction.sampling_type == SamplingType.TOPK:
                k = instruction.k
                top_k = torch.topk(decoder_out[:,-1,:], k) # ([values], [indices])
                probs = top_k[0].softmax(dim=-1)
                indices_of_k = torch.multinomial(
                    input=probs,
                    num_samples=1,
                    replacement=True
                )
                nxt = torch.gather(top_k[1], dim=-1, index=indices_of_k)
                log_p_nxt = torch.log(torch.gather(probs, dim=-1, index=indices_of_k))

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
                        num_samples=1,
                        replacement=True
                    )
                    log_p_nxt = torch.log(torch.gather(norm_probs, dim=-1, index=indices_of_p))
                    nxt = torch.gather(indices, dim=-1, index=indices_of_p)
                else:
                    nxt = torch.argmax(probs, dim=-1, keepdim=True)
                    log_p_nxt = nxt * 0
            
            else:
                raise Exception(f"Invalid sampling type {instruction.sampling_type}")

            if tgt is None:
                seq = torch.cat([nxt], dim=1)
            else:
                seq = torch.cat([tgt, nxt], dim=1)
            
            return seq

        ### BEAM SEARCH
        elif instruction.decode_type == DecodeType.BEAM:
            if tgt is None:
                (B, length, hypo) = (x.shape[0], 1, 1)
                decoder_out = self.forward(x, tgt)
            else:
                (B, length, hypo) = tgt.shape
                decoder_out = self.forward(
                    x.repeat((instruction.beam_size, 1, 1)),
                    tgt.permute(0, 2, 1).flatten(end_dim=1)
                )
                # decoder_out = torch.zeros(B * instruction.beam_size, length, 3032).to(self.device)
            # decoder_out : (batch_size * hypotheses, seq_len, logits)
            if instruction.sampling_type == SamplingType.MULTINOMIAL:
                probs = decoder_out[:,-1,:].softmax(dim=-1)
                log_p_nxt = torch.log(probs)
                nxt = torch.multinomial(
                    input=probs,
                    num_samples=instruction.beam_size,
                    replacement=True
                ) # Default ancestral
            
            elif instruction.sampling_type == SamplingType.TEMPERATURE:
                probs = (decoder_out[:,-1,:] / instruction.temp).softmax(dim=-1)
                log_p_nxt = torch.log(probs)
                nxt = torch.multinomial(
                    input=probs,
                    num_samples=instruction.beam_size,
                    replacement=True
                )
            
            elif instruction.sampling_type == SamplingType.GREEDY:
                log_p_nxt = decoder_out[:,-1,0] * 0
                nxt = torch.argmax(decoder_out[:,-1,:], dim=-1, keepdim=True) # Pointless
            
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
                log_p_nxt = torch.log(torch.gather(probs, dim=-1, index=indices_of_k))

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
                    log_p_nxt = torch.log(torch.gather(norm_probs, dim=-1, index=indices_of_p))
                    nxt = torch.gather(indices, dim=-1, index=indices_of_p)
                else:
                    nxt = torch.argmax(probs, dim=-1, keepdim=True)
                    log_p_nxt = nxt * 0
            
            else:
                raise Exception(f"Invalid sampling type {instruction.sampling_type}")

            # Expand hypotheses
            nxt_log_prob = torch.gather(log_p_nxt, dim=-1, index=nxt).unflatten(dim=0, sizes=(B, hypo))
            new_scores = (scores.unsqueeze(dim=-2) + nxt_log_prob).flatten(start_dim=1) # (batch_size, num_hypo * beam_size)
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
        if instruction.decode_type == DecodeType.BEAM:
            scores = torch.zeros((x.shape[0], instruction.beam_size)).to(self.device)
        else:
            scores = torch.zeros((x.shape[0],)).to(self.device)
        seq = self._step(x, tgt, instruction, scores)
        continue_samples = torch.ones(seq[:,-1].shape).to(self.device)

        while not torch.all(continue_samples == 0) and seq.shape[1] < instruction.max_seq_len:
            seq = self._step(x, seq, instruction, scores)
            continue_samples = continue_samples * (seq[:,-1] != self.eos_token[:x.shape[0]])

        if instruction.decode_type == DecodeType.BEAM:
            # Best sequence
            seq = torch.gather(
                input=seq,
                dim=-1,
                index=scores.topk(k=1, dim=-1)[1][:,None].expand(-1, seq.shape[1], -1),
            )
        
        return seq

    def forward(self, x : torch.Tensor, tgt : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded sequence from the encoder
        tgt (torch.Tensor): the unencoded, unembedded target sequence to pass directly into the decoder
                          in order to generate the next token

        Returns:
        --------
        torch.Tensor: the logits for next token selection (batch_size, vocab_size)
        '''
        # x : (batch_size, seq_len, vocab_size)
        if tgt is not None and tgt.shape[1] != 0:
            embeddings = self.embedder(torch.cat([self.sos_token[:x.shape[0]], tgt], dim=1))
            embeddings += self.pos_embed[:,:tgt.shape[1]+1,:]
        else:
            embeddings = self.embedder(self.sos_token[:x.shape[0]])
            embeddings += self.pos_embed[:,:1,:]
        embeddings = self.dropout(embeddings)
        for module in self.transformer_decoder_layers:
            embeddings = module(x, embeddings)
        return self.token_space(embeddings)


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
    #     decoder_out = self.decoder(encoder_out, tgt)[:,-1,:]
    #     nxt = torch.multinomial(decoder_out, 1) # Default ancestral
    #     seq = torch.cat([nxt], dim=1)
    #     continue_samples = torch.ones(nxt.shape) * (nxt != eos_token)

    #     while not torch.all(continue_samples == 0) or seq.shape[1] > 9:
    #         decoder_out = self.decoder(encoder_out, seq)[:,-1,:]
    #         nxt = torch.multinomial(decoder_out, 1) # Default ancestral
    #         seq = torch.cat([seq, nxt], dim=1)
    #         continue_samples = continue_samples * (nxt != eos_token)
        
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