import torch
import torch.nn as nn
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerEncoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the partially encoded source sequence from the previous layer
        '''
        mhsa_out, _ = self.MHA(x, x, x)
        x = self.norm_1(self.dropout_1(mhsa_out) + x)
        ff_out = self.ff(x)
        x = self.norm_2(ff_out + x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float = 0.1) -> nn.Module:
        super(TransformerDecoderLayer, self).__init__()

        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.norm_3 = nn.LayerNorm(embedding_dim)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.MaskedMHSA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.MHA = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded source sequence from the encoder
        y (torch.Tensor): the target sequence upon which to generate the next token
        '''
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(y.shape[1], y.device)
        masked_mhsa_out, _ = self.MaskedMHSA(y, y, y, attn_mask=causal_mask, is_causal=True, need_weights=False)
        y = self.norm_1(self.dropout_1(masked_mhsa_out) + y)
        mha_out, _ = self.MHA(y, x, x, need_weights=False)
        y = self.norm_2(self.dropout_2(mha_out) + y)
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
        # Learned position embeddings
        # self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True)

        # Sinusoidal position embeddings + learned map
        d = embedding_dim # Dimension of position embedding
        embedded_frequencies = torch.Tensor(torch.pow(torch.Tensor([0.0001]), 2 / d * torch.ceil(torch.linspace(1, d, d) / 2))).to(device)
        sin_hot = (torch.linspace(1, d, d) % 2 == 0).to(device)
        cos_hot = (torch.linspace(1, d, d) % 2 == 1).to(device)
        t = torch.linspace(0, 9999, 10000).to(device)
        self.pos_embed = (torch.sin(torch.outer(t, embedded_frequencies)) * sin_hot + torch.cos(torch.outer(t, embedded_frequencies)) * cos_hot).unsqueeze(0)
        self.pos_map = nn.Linear(d, embedding_dim)
        
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
        embeddings += self.pos_map(self.pos_embed[:,:src.shape[1]+1,:])
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

        # Learned position embeddings
        # self.pos_embed = nn.Parameter(torch.zeros(1, 10000, embedding_dim), requires_grad=True)

        # Sinusoidal position embeddings + learned map
        d = embedding_dim # Dimension of position embedding
        embedded_frequencies = torch.Tensor(torch.pow(torch.Tensor([0.0001]), 2 / d * torch.ceil(torch.linspace(1, d, d) / 2))).to(device)
        sin_hot = (torch.linspace(1, d, d) % 2 == 0).to(device)
        cos_hot = (torch.linspace(1, d, d) % 2 == 1).to(device)
        t = torch.linspace(0, 9999, 10000).to(device)
        self.pos_embed = (torch.sin(torch.outer(t, embedded_frequencies)) * sin_hot + torch.cos(torch.outer(t, embedded_frequencies)) * cos_hot).unsqueeze(0)
        self.pos_map = nn.Linear(d, embedding_dim)

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
        torch.Tensor: the logits for next token selection (batch_size, vocab_size)
        '''
        # x : (batch_size, seq_len, vocab_size)
        if tgt is not None and tgt.shape[1] != 0:
            embeddings = self.embedder(torch.cat([self.sos_token[:x.shape[0]], tgt], dim=1))
            # embeddings += self.pos_embed[:,:tgt.shape[1]+1,:]
            embeddings += self.pos_map(self.pos_embed[:,:tgt.shape[1]+1,:])
        else:
            embeddings = self.embedder(self.sos_token[:x.shape[0]])
            # embeddings += self.pos_embed[:,:1,:]
            embeddings += self.pos_map(self.pos_embed[:,:1,:])
        embeddings = self.dropout(embeddings)
        for module in self.transformer_decoder_layers:
            embeddings = module(x, embeddings)
        return self.token_space(embeddings)

    def pos_embedding(self, t : torch.Tensor):
        # sine embedding
        return torch.sin(torch.outer(t, self.embedded_frequencies)) * self.sin_hot + torch.cos(torch.outer(t, self.embedded_frequencies)) * self.cos_hot

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        return self.token_space(self.embedder(x))

    @torch.no_grad()
    def _step(self, x : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None,
                    scores : torch.Tensor = None, continue_samples : torch.Tensor = None) -> torch.Tensor:
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
                    nxt = torch.gather(indices, dim=-1, index=indices_of_p)
                else:
                    nxt = torch.argmax(probs, dim=-1, keepdim=True)
            
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
                decoder_out = self.forward(
                    x, # (batch_size, src_seq_len, embed_dim)
                    tgt, # None
                )
            else:
                (B, length, hypo) = tgt.shape
                decoder_out = self.forward(
                    x.repeat((instruction.beam_size, 1, 1)),
                    tgt.permute(0, 2, 1).flatten(end_dim=1)
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
                scores = torch.zeros((x.shape[0], instruction.beam_size)).to(self.device)
            else:
                scores = torch.zeros((x.shape[0],)).to(self.device)
            
            continue_samples = torch.ones(x.shape[0],).to(self.device)
            seq = self._step(x, tgt, instruction, scores, continue_samples)

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
            seq = self._step(x, seq, instruction, scores, continue_samples)

            if instruction.decode_type == DecodeType.BEAM:
                continue_samples = continue_samples * torch.all(seq[:,-1] != self.eos_token[:x.shape[0]], dim=1)
            else:
                continue_samples = continue_samples * (seq[:,-1] != self.eos_token[:x.shape[0]])

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
                outs.append(seq[b_idx,:index_of_eos,max_idx])
        else:
            outs = []
            for b_idx in range(x.shape[0]):
                if self.eos_token[0] not in seq[b_idx,:]:
                    print(f"Decoded sequence does not have an EOS token for batch index {b_idx}. Passing.")
                    # continue
                    index_of_eos = seq.shape[1]
                else:
                    index_of_eos = (seq[b_idx,:] == self.eos_token[0]).nonzero(as_tuple=True)[0][0]
                outs.append(seq[b_idx,:index_of_eos])
        
        return outs


class FontModel(nn.Module):
    def __init__(self, num_enc_layers : int, num_dec_layers : int, vocab_size : int, embedding_dim : int, num_heads : int, ff_dim : int, dropout_rate : float, device : torch.device) -> nn.Module:
        super(FontModel, self).__init__()

        self.embedder = nn.Embedding(vocab_size, embedding_dim)
        
        self.encoder = TransformerEncoder(
            num_layers=num_enc_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            embedder=self.embedder,
            dropout_rate=dropout_rate,
            device=device
        )
        self.decoder = TransformerDecoder(
            num_layers=num_dec_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            embedder=self.embedder,
            dropout_rate=dropout_rate,
            device=device
        )

    def identity_embeddings(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Used for learning useful embeddings by training an identity function through a bottleneck.
        The embeddings learned will be used in the regular forward pass after pretraining.
        '''
        ### If using custom transformer
        return self.encoder.identity_embeddings(x)

    def decode(self, x : torch.Tensor, tgt : torch.Tensor = None, instruction : DecodeInstruction = None) -> torch.Tensor:
        '''
        Parameters:
        -----------
        x (torch.Tensor): the encoded source CLS token (batch_size, 1, embed_dim)
        tgt (torch.Tensor): the target sequence to pass directly into the decoder
                          in order to generate the next token (leave None if generate from start)
        instruction (DecodeInstruction): the instruction for how to decode

        Returns:
        --------
        torch.Tensor: the generated sequence (batch_size, max_seq_len, vocab_size)
        '''
        # src : (batch_size, seq_len, vocab_size)
        return self.decoder.decode(x, tgt, instruction)
        

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
        x = self.encoder(src)[:,0:1,:]
        decoder_out = self.decoder(x, tgt)
        return decoder_out


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, cond_dimension, proj_dimension):
        super(CrossAttentionBlock, self).__init__()

        self.Wq = nn.Linear(channels, proj_dimension, bias=False)
        self.Wk = nn.Linear(cond_dimension, proj_dimension, bias=False)
        self.Wv = nn.Linear(cond_dimension, proj_dimension, bias=False)
        self.out_proj = nn.Linear(proj_dimension, channels, bias=False)
        self.scale = 1 / np.sqrt(channels)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, y):
        ''' x: Tensor of shape (batch_size, num_channels, num_rows, num_cols)
            y: Tensor of shape (batch_size, num_conditions, condition_dimension)
        '''
        # x (batch, channels, r, c) --> moveaxis, flatten, Wq --> Q (batch, r*c, proj_dim)
        # cond (batch, num_cond, cond_dim) --> Wk/Wv --> K/V (batch, num_cond, proj_dim)
        # K.swapaxes (batch, proj_dim, num_cond)
        # Q @ K.swapaxes (batch, r*c, num_cond)
        # Q @ K.swapaxes * V (batch, r*c, proj_dim) --> out_proj, moveaxis, reshape --> (batch, channels, r, c)
        if y is None:
            return x
        Q = self.Wq(x.moveaxis(-3, -1).flatten(-3, -2))
        KT = self.Wk(y).swapaxes(-2, -1)
        V = self.Wv(y)
        attn = self.out_proj(self.softmax(Q @ KT * self.scale) @ V).moveaxis(-1, -2).unflatten(-1, (x.shape[-2:]))
        return x + attn


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetDoubleConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=conv_map['kernel'], stride=conv_map['stride'], padding=conv_map['padding'], groups=1, bias=False, dilation=conv_map['dilation']),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2)
        )
        self.time_map = nn.Sequential(
            nn.Linear(time_dimension, out_channels, bias=False),
            nn.LeakyReLU(0.2)
        )
        # self.cond_map = nn.Sequential(
        #     nn.Linear(cond_dimension, out_channels, bias=False),
        #     nn.LeakyReLU(0.2)
        # )
        self.cond_map = CrossAttentionBlock(out_channels, cond_dimension, 6)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, time_embedding, y):
        dc_new = self.conv2(self.cond_map(self.conv1(x), y) + self.time_map(time_embedding)[:,:,None,None])
        return self.batch_norm(self.act(dc_new + self.res_conv(x)))


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetDownBlock, self).__init__()

        self.double_conv = UNetDoubleConv(in_channels, out_channels, time_dimension, cond_dimension, conv_map)
        self.pool = nn.Conv1d(out_channels, out_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        # self.pool = nn.MaxPool1d(2)

    def forward(self, x, time_embedding, y):
        conved = self.double_conv(x, time_embedding, y)
        return conved, self.pool(conved)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dimension, cond_dimension, conv_map):
        super(UNetUpBlock, self).__init__()

        self.unpool = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=conv_map['down_up_kernel_and_stride'], stride=conv_map['down_up_kernel_and_stride'])
        self.double_conv = UNetDoubleConv(2 * in_channels, out_channels, time_dimension, cond_dimension, conv_map)

        self.attention_conv = nn.Conv1d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(in_channels)

    def forward(self, x, x_prior, time_embedding, y):
        return self.double_conv(torch.cat((x_prior, self.unpool(x)), dim=1), time_embedding, y)


class UNet(nn.Module):
    def __init__(self, in_channels, time_dimension, cond_dimension, conv_map):
        super(UNet, self).__init__()

        self.down1 = UNetDownBlock(in_channels, 64, time_dimension, cond_dimension, conv_map)
        self.down2 = UNetDownBlock(64, 128, time_dimension, cond_dimension, conv_map)
        self.down3 = UNetDownBlock(128, 256, time_dimension, cond_dimension, conv_map)
        self.down4 = UNetDownBlock(256, 512, time_dimension, cond_dimension, conv_map)
        self.down5 = UNetDownBlock(512, 1024, time_dimension, cond_dimension, conv_map)
        self.down6 = UNetDownBlock(1024, 2048, time_dimension, cond_dimension, conv_map)
        self.layer7 = UNetDoubleConv(2048, 2048, time_dimension, cond_dimension, conv_map)
        self.up6 = UNetUpBlock(2048, 1024, time_dimension, cond_dimension, conv_map)
        self.up5 = UNetUpBlock(1024, 512, time_dimension, cond_dimension, conv_map)
        self.up4 = UNetUpBlock(512, 256, time_dimension, cond_dimension, conv_map)
        self.up3 = UNetUpBlock(256, 128, time_dimension, cond_dimension, conv_map)
        self.up2 = UNetUpBlock(128, 64, time_dimension, cond_dimension, conv_map)
        self.up1 = UNetUpBlock(64, in_channels, time_dimension, cond_dimension, conv_map)
        self.last = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
            
    def forward(self, x, time_embedding, y):
        x1, x_run = self.down1(x, time_embedding, y)
        x2, x_run = self.down2(x_run, time_embedding, y)
        x3, x_run = self.down3(x_run, time_embedding, y)
        x4, x_run = self.down4(x_run, time_embedding, y)
        x5, x_run = self.down5(x_run, time_embedding, y)
        x6, x_run = self.down6(x_run, time_embedding, y)
        x_run = self.layer7(x_run, time_embedding, y)
        x_run = self.up6(x_run, x6, time_embedding, y)
        x_run = self.up5(x_run, x5, time_embedding, y)
        x_run = self.up4(x_run, x4, time_embedding, y)
        x_run = self.up3(x_run, x3, time_embedding, y)
        x_run = self.up2(x_run, x2, time_embedding, y)
        return self.last(self.up1(x_run, x1, time_embedding, y))


class CLSDiffusionModel(nn.Module):
    def __init__(self, depth : int, label_dim : int, num_classes : int, conv_map : dict, device : torch.device):
        super(CLSDiffusionModel, self).__init__()

        self.device = device
        self.alphas = torch.linspace(0.9999, 0.98, depth+1).to(device)
        self.alpha_bars = torch.Tensor([torch.prod(self.alphas[:i+1]) for i in range(depth+1)]).to(device)

        d = 100 # Dimension of time embedding
        self.embedded_frequencies = torch.pow(torch.Tensor([0.0001]), 2 / d * torch.ceil(torch.linspace(1, d, d) / 2)).to(device)
        self.sin_hot = (torch.linspace(1, d, d) % 2 == 0).to(device)
        self.cos_hot = (torch.linspace(1, d, d) % 2 == 1).to(device)

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

        self.noise_pred = UNet(1, d, c, conv_map).to(device)

    def reparameterize(self, mean, var):
        eps = torch.randn_like(mean).to(self.device)
        return mean + torch.sqrt(var) * eps, eps

    def diffuse(self, x0, t):
        # mean, var for x_t sampled along q(x_t | x_0)
        mean = torch.sqrt(self.alpha_bars[t])[:,None,None] * x0
        var = (1 - self.alpha_bars[t])[:,None,None]
        x_t, eps = self.reparameterize(mean, var)
        return x_t, eps

    def predict_noise(self, x_t, t, y):
        time = self.time_embedding(t)
        if y is None:
            cond_emb = None
        elif self.num_classes is None:
            cond_emb = self.cond_embedding(y)
        else:
            cond_emb = self.cond_embedding(y.long())
        predicted_noise = self.noise_pred(x_t, time, cond_emb)

        return predicted_noise

    def sample(self, x_t, t, y, cfg_coeff=3):
        # x_{t-1} sampled along p(x_{t-1} | x_t)
        predicted_noise = self.forward(x_t, t, y)
        if cfg_coeff > 0:
            unconditional_predicted_noise = self.predict_noise(x_t, t, None)
            predicted_noise = torch.lerp(predicted_noise, unconditional_predicted_noise, -cfg_coeff)

        # DDPM
        mean = 1 / torch.sqrt(self.alphas[t])[:,None,None] * (x_t - (1 - self.alphas[t])[:,None,None] / torch.sqrt(1 - self.alpha_bars[t])[:,None,None] * predicted_noise)
        var = ((1 - self.alphas[t]) * (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]))[:,None,None]
        eps = torch.randn_like(mean).to(self.device) * (t > 1)
        return mean + torch.sqrt(var) * eps