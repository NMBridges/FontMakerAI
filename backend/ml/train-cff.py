#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from config import conv_map, device, operators

from fontmodel import (FontModel, DecodeInstruction, DecodeType, SamplingType, TransformerScheduler)
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from performance import PerformanceMetrics
from tablelist_utils import numbers_first, make_non_cumulative


# ### Training arguments

# In[ ]:


print(f"Executing train-cff.ipynb on {device}...\n-----------------------------")

args = {
    "load_model": True,
    "train_transformer": True,
    "min_number": -500,
    "max_number": 500,
    "max_seq_len": 5040,
    "num_layers": 12,
    "embedding_dim": 1024,
    "num_heads": 16,
    "ff_dim": 4096,
    "use_wandb": True,
    "epochs": 15,
    "batch_size": 32,
    "batch_accumulate": 4,
    "lr": 6e-4,
    "dropout_rate": 0.2,
    "weight_decay": 1e-1,
    "gradient_clip": True,
    "gradient_clip_val": 1.0,
    "label_smoothing": 0.001,
    "sample_every": 1,
    "use_scheduler": True,
    "scheduler_warmup_steps": 2000,
    "data_type": torch.bfloat16,
    "vae_beta": 1e-1,
    "vae_epochs": 10,
    "vae_lr": 1e-2,
    "vae_weight_decay": 1e-5,
    "freeze_embeddings": False,
    "use_pretrained_embeddings": False,
    "pretrain_embeddings": False,
    "pretrain_epochs": 1,
    "pretrain_batch_size": 128,
    "pretrain_lr": 4e-3,
    "pretrain_use_scheduler": True,
    "pretrain_scheduler_warmup_steps": 3000,
    "use_pretrained_vit_encoder": False,
    "pretrain_vit_encoder": False,
    "pretrain_vit_encoder_epochs": 1,
    "pretrain_vit_encoder_batch_size": 128,
    "pretrain_vit_encoder_batch_accumulate": 1,
    "pretrain_vit_encoder_lr": 1e-3,
    "pretrain_vit_encoder_weight_decay": 1e-3,
    "pretrain_vit_encoder_use_scheduler": True,
    "pretrain_vit_encoder_scheduler_warmup_steps": 1500,
    "post_train": False,
    "post_train_epochs": 1,
    "post_train_batch_size": 32,
    "post_train_lr": 6e-4,
    "post_train_kl_penalty": 0.05,
    "post_train_use_scheduler": True,
    "post_train_scheduler_warmup_steps": 2000,
}

print("Training hyperparameters:")
pprint(args)


# ### Tokenization Scheme

# In[ ]:


pad_token = "<PAD>"
sos_token = "<SOS>"
eos_token = "<EOS>"
tokenizer = Tokenizer(
    min_number=args['min_number'],
    max_number=args['max_number'],
    possible_operators=operators,
    pad_token=pad_token,
    sos_token=sos_token,
    eos_token=eos_token
)
cumulative = True
vocab_size = tokenizer.num_tokens


# ### Sampling Scheme

# In[ ]:


decode_instr = DecodeInstruction( # NOTE: doesn't matter unless loading from .config.txt fails
    DecodeType.ANCESTRAL,
    SamplingType.GREEDY,
    max_seq_len=args['max_seq_len'],
    k=5,
    p=0,
    temp=0,
    beam_size=6,
)


# ### Create model

# In[ ]:


if args['load_model']:
    model_pre = torch.load(f'models/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', map_location=device, weights_only=False).to(device)
else:
    model_pre = FontModel(
        num_enc_layers=args['num_layers'],
        num_dec_layers=args['num_layers'],
        vocab_size=vocab_size,
        embedding_dim=args['embedding_dim'],
        num_heads=args['num_heads'],
        ff_dim=args['ff_dim'],
        dropout_rate=args['dropout_rate'],
        max_seq_len=args['max_seq_len'],
        device=device
    ).to(device, dtype=args['data_type'])
model = torch.compile(model_pre)


# In[ ]:


def count_params(modela):
    model_parameters = filter(lambda p: p.requires_grad, modela.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of trainable parameters: {params}")
    return params

x = count_params(model)
y = count_params(model.encoder)
z = count_params(model.decoder)
w = count_params(model.encoder.embedder)
v = count_params(model.decoder.embedder)


# ### Training tools

# In[ ]:


# Parameters (tentative):
# FontModel: embedder (DON'T APPLY WEIGHT DECAY)
# TransformerDecoder: transformer_decoder_layers (DON'T APPLY WEIGHT DECAY TO RMSNORM), command_encoder, command_decoder, norm_final (DON'T APPLY WEIGHT DECAY)
# TransformerEncoder: transformer_encoder_layers (DON'T APPLY WEIGHT DECAY TO RMSNORM), embedder (custom),pos_embed, norm_final (DON'T APPLY WEIGHT DECAY)

# We don't want to apply weight decay to layer norms and embeddings
no_weight_decay_params = [x for x in model.decoder.embedder.parameters() if x.requires_grad]
no_weight_decay_params += [x for x in model.decoder.inverse_embedder.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.decoder.transformer_decoder_layers.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
no_weight_decay_params += [x for x in model.decoder.norm_final.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
no_weight_decay_params += [x for x in model.encoder.norm_final.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
no_weight_decay_params += [x for x in model.decoder.command_encoder.parameters() if x.requires_grad]
no_weight_decay_params += [x for x in model.decoder.command_decoder.parameters() if x.requires_grad]
no_weight_decay_params += [x for x in model.decoder.command_decoder_2a.parameters() if x.requires_grad]
no_weight_decay_params += [x for x in model.decoder.command_decoder_2b.parameters() if x.requires_grad]
# no_weight_decay_params += [x for x in model.decoder.command_decoder_1.parameters() if x.requires_grad]
# no_weight_decay_params += [x for x in model.decoder.command_decoder_2.parameters() if x.requires_grad]
# no_weight_decay_params += [x for x in model.decoder.W_cn.parameters() if x.requires_grad]
# no_weight_decay_params += [x for x in model.decoder.W_cnb.parameters() if x.requires_grad]

weight_decay_params = [x for name, x in model.decoder.transformer_decoder_layers.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for x in model.encoder.pos_embed.parameters() if x.requires_grad]

vit_encoder_params_nwd = [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad]# and ('norm' in name or 'bias' in name)]
# vit_encoder_params_nwd += [x for name, x in model.encoder.pretrain_reverse_ae.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
vit_encoder_params_nwd += [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
vit_encoder_params_nwd += [x for x in model.encoder.norm_final.parameters() if x.requires_grad]
# vit_encoder_params_wd = [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
vit_encoder_params_nwd += [x for name, x in model.encoder.pretrain_reverse_ae.named_parameters() if x.requires_grad]# and 'norm' not in name and 'bias' not in name]
vit_encoder_params_wd = [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
vit_encoder_params_wd += [x for x in model.encoder.pos_embed.parameters() if x.requires_grad]

optimizer = torch.optim.AdamW(
    [
       {'params': weight_decay_params, 'weight_decay': args['weight_decay']},
       {'params': no_weight_decay_params, 'weight_decay': args['weight_decay']}
    ],
    betas=(0.9, 0.95),
    lr=args['lr'] 
)

pretrain_optimizer = torch.optim.AdamW(no_weight_decay_params, weight_decay=0.0, betas=(0.9, 0.95), lr=args['pretrain_lr'])
posttrain_optimizer = torch.optim.AdamW(
    [
       {'params': weight_decay_params, 'weight_decay': 0.0},
       {'params': no_weight_decay_params, 'weight_decay': 0.0}
    ],
    betas=(0.9, 0.95),
    lr=args['post_train_lr']
)

pretrain_vit_encoder_optimizer = torch.optim.AdamW(
    [
        {'params': vit_encoder_params_wd, 'weight_decay': args['pretrain_vit_encoder_weight_decay']},
        {'params': vit_encoder_params_nwd, 'weight_decay': 0.0},
    ],
    betas=(0.9, 0.95),
    lr=args['pretrain_vit_encoder_lr']
)

max_len = 33928
num_glyphs = 26
step_every = 1

if args['use_scheduler']:
    # scheduler = TransformerScheduler(
    #     optimizer=optimizer,
    #     dim_embed=args['embedding_dim'],
    #     warmup_steps=args['scheduler_warmup_steps']
    # )
    batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['batch_size'] + 0.5)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * (batches_per_epoch // args['batch_accumulate']), eta_min=1e-5)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['scheduler_warmup_steps'])
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)

if args['pretrain_use_scheduler']:
    pretrain_batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['pretrain_batch_size'] + 0.5)
    pretrain_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=args['pretrain_epochs'] * pretrain_batches_per_epoch, eta_min=1e-5)
    pretrain_scheduler2 = torch.optim.lr_scheduler.LinearLR(pretrain_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['pretrain_scheduler_warmup_steps'])
    pretrain_scheduler = torch.optim.lr_scheduler.ChainedScheduler([pretrain_scheduler1, pretrain_scheduler2], optimizer=pretrain_optimizer)

if args['post_train_use_scheduler']:
    posttrain_batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['post_train_batch_size'] + 0.5)
    posttrain_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(posttrain_optimizer, T_max=args['post_train_epochs'] * posttrain_batches_per_epoch, eta_min=1e-5)
    posttrain_scheduler2 = torch.optim.lr_scheduler.LinearLR(posttrain_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['post_train_scheduler_warmup_steps'])
    posttrain_scheduler = torch.optim.lr_scheduler.ChainedScheduler([posttrain_scheduler1, posttrain_scheduler2], optimizer=posttrain_optimizer)

if args['pretrain_vit_encoder_use_scheduler']:
    pretrain_vit_encoder_batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['pretrain_vit_encoder_batch_size'] + 0.5)
    pretrain_vit_encoder_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_vit_encoder_optimizer, T_max=args['pretrain_vit_encoder_epochs'] * (pretrain_vit_encoder_batches_per_epoch // args['pretrain_vit_encoder_batch_accumulate']), eta_min=1e-5)
    pretrain_vit_encoder_scheduler2 = torch.optim.lr_scheduler.LinearLR(pretrain_vit_encoder_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['pretrain_vit_encoder_scheduler_warmup_steps'])
    pretrain_vit_encoder_scheduler = torch.optim.lr_scheduler.ChainedScheduler([pretrain_vit_encoder_scheduler1, pretrain_vit_encoder_scheduler2], optimizer=pretrain_vit_encoder_optimizer)

dataset_name = f"basic-33928allchars_centered_scaled_sorted_filtered{'_cumulative' if cumulative else ''}_padded"
train_start, train_end = 0, int(0.95 * max_len) * num_glyphs
test_start, test_end = train_end, max_len * num_glyphs
# max_len = 5
# train_start, train_end = 0, 26*max_len
# test_start, test_end = 0, 26*max_len
cff_dataset = torch.load(f'./{dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
cff_dataset_test = torch.load(f'./{dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
im_dataset_name = "basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)"
im_dataset = torch.load(f'./{im_dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
im_dataset_test = torch.load(f'./{im_dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
cff_train_tensor_dataset = TensorDataset(cff_dataset, im_dataset)
cff_train_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['batch_size'], shuffle=True)
cff_pretrain_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['pretrain_batch_size'], shuffle=True)
cff_pretrain_vit_encoder_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['pretrain_vit_encoder_batch_size'], shuffle=True)
cff_posttrain_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['post_train_batch_size'], shuffle=True)
cff_test_tensor_dataset = TensorDataset(cff_dataset_test, im_dataset_test)
cff_test_dataloader = DataLoader(cff_test_tensor_dataset, batch_size=args['batch_size'], shuffle=True)


# In[ ]:


if args['use_wandb']:
    wandb.init(
        project="project-typeface",
        config={
            "model_type": "Autoregressive CFF",
            **args
        }
    )


# ### Useful Loss Functions

# In[ ]:


loss_fn = torch.nn.CrossEntropyLoss(
    reduction='sum',
    ignore_index=tokenizer[pad_token],
    label_smoothing=args['label_smoothing']
)
test_loss_fn = torch.nn.CrossEntropyLoss(
    reduction='sum',
    ignore_index=tokenizer[pad_token],
    label_smoothing=0.0
)
def recon_loss(a, b):
    return torch.pow((a - b), 2).sum()

def mse_loss(out_scaled, target_unscaled):
    # return torch.pow(((target_unscaled - 32) / (args['max_number'] - args['min_number']) * 2 - 1 - out_scaled) * (target_unscaled != 0), 2).sum()
    return (((target_unscaled - 32) / (args['max_number'] - args['min_number']) * 2 - 1 - out_scaled) * (target_unscaled != 0)).abs().sum()

def kl_loss_fn(mu, logvar):
    return 0.5 * ((torch.pow(mu, 2) + logvar.exp() - logvar - 1)).sum()

curve_width = 7
curve_coeff = 1.0
first_numeric_idx = tokenizer.special_tokens_len + len(tokenizer.possible_operators)
zero_mask = (torch.ones((1,1,1,vocab_size)) * (torch.arange(0,vocab_size, dtype=torch.int32) >= first_numeric_idx)).to(device)
hlf = curve_width // 2
offset = torch.arange(-hlf, hlf+1, dtype=torch.int32)[None,None,:].to(device) # (1, 1, curve_width)
neg_offset_exp = (-offset.abs().unsqueeze(-1) * 7.0 * curve_coeff / curve_width).exp()
kl_loss = torch.nn.KLDivLoss(reduction='sum', log_target=False)
log_kl_loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)

def numeric_mse_loss(output : torch.Tensor, targets : torch.Tensor):
    # targets : (batch_size, seq_len)
    with torch.no_grad():
        is_numeric_tgt = (targets >= first_numeric_idx).unsqueeze(-1) # (batch_size, seq_len, 1)
        is_non_padding = (targets > 0).unsqueeze(-1) # (batch_size, seq_len, 1)
        # TODO: try removing the calculations from GPU and putting on CPU
        token_count = targets.shape[0] * targets.shape[1]
        arrng = torch.arange(0, token_count, dtype=torch.int32)
        batch_indices = torch.floor_divide(arrng, targets.shape[1]).unsqueeze(0).to(device)
        sequence_indices = torch.remainder(arrng, targets.shape[1]).unsqueeze(0).to(device)
        token_indices = targets.flatten().unsqueeze(0)
        single_tgt = torch.sparse_coo_tensor(
            indices=torch.cat([batch_indices, sequence_indices, token_indices], dim=0),
            values=torch.ones(token_count,).to(device),
            size=(targets.shape[0], targets.shape[1], tokenizer.num_tokens)
        ) # (batch_size, seq_len, vocab_size)
        
        arrng = torch.arange(0, token_count * curve_width, dtype=torch.int32)
        batch_indices = torch.floor_divide(arrng, targets.shape[1] * curve_width).unsqueeze(0).to(device)
        sequence_indices = torch.floor_divide(torch.remainder(arrng, targets.shape[1] * curve_width), curve_width).unsqueeze(0).to(device)
        curve_indices = torch.remainder(arrng, curve_width).unsqueeze(0).to(device)
        token_indices = torch.clip(targets.unsqueeze(-1) + offset, min=1, max=tokenizer.num_tokens-1).flatten().unsqueeze(0)
        multi_tgt = torch.sparse_coo_tensor(
            indices=torch.cat([batch_indices, sequence_indices, curve_indices, token_indices], dim=0),
            values=torch.ones(token_count * curve_width,).to(device),
            size=(targets.shape[0], targets.shape[1], curve_width, tokenizer.num_tokens)
        ) # (batch_size, seq_len, curve_width, vocab_size)
        multi_tgt = multi_tgt * neg_offset_exp * zero_mask
        multi_tgt = (multi_tgt.sum(dim=-2).to_dense() / (1e-10 + multi_tgt.sum(dim=(-2,-1)).unsqueeze(-1).to_dense())).to_sparse() # (batch_size, seq_len, vocab_size)
        tgt_dist = is_numeric_tgt * multi_tgt + (~is_numeric_tgt) * (is_non_padding) * single_tgt # (batch_size, seq_len, vocab_size)

    return kl_loss(output.log_softmax(dim=-1), tgt_dist.to_dense())


# ### Useful Decoding Function

# In[ ]:


def decode(epoch : int, batch_idx : int):
    with open('./fontmakerai/.config.txt', 'r') as cf:
        lines = cf.readlines()
        if len(lines) != 7:
            print(f"Not decoding this iteration; .config.txt has wrong number of lines ({len(lines)})")
            return
        else:
            decode_instr = DecodeInstruction(
                decode_type=DecodeType[lines[0].split("=")[-1].split(".")[-1].strip()],
                sampling_type=SamplingType[lines[1].split("=")[-1].split(".")[-1].strip()],
                max_seq_len=int(lines[2].split("=")[-1].strip()),
                k=int(lines[3].split("=")[-1].strip()),
                p=float(lines[4].split("=")[-1].strip()),
                temp=float(lines[5].split("=")[-1].strip()),
                beam_size=int(lines[6].split("=")[-1].strip())
            )

    model.eval()
    with torch.no_grad():
        try:
            flag = True
            idx = np.random.randint(0, im_dataset_test.shape[0])
            im = im_dataset_test[idx:idx+1].to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            sequence = model.decode(im, None, decode_instr)[0].cpu().detach().numpy().flatten()
            torch.cuda.empty_cache()
            # sequence = cff_train_tensor_dataset[0:1][0][0].cpu().detach().numpy().flatten()#.to(device)
            if len(sequence) == decode_instr.max_seq_len:
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence] + ['endchar']
            else:
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

            print("Before:", toks)
            with open(f"./fontmakerai/training_images/{epoch+1}_{batch_idx+1}.txt", 'w') as f:
                j_str = '\', \''
                f.write(f"Before: ['{j_str.join([str(x) for x in toks])}']\n\n")
                f.write(f"decode_instr:")
                for k, v in decode_instr.__dict__.items():
                    f.write(f"\n\t{k}={v}")
                f.write("\n")
            toks = [tok for tok in toks if tok != '<PAD2>' and tok != '<PAD>']
            if cumulative:
                toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
                # toks = make_non_cumulative(toks, tokenizer)
            else:
                toks = numbers_first(toks, tokenizer, return_string=False)
                # toks = toks
            print("After:", toks)
            viz = Visualizer(toks)
            with open(f"./fontmakerai/training_images/{epoch+1}_{batch_idx+1}.txt", 'a', newline='\n') as f:
                j_str = '\', \''
                f.write(f"After: ['{j_str.join([str(x) for x in toks])}']")

            if toks[2] != "rmoveto" and toks[3] != "rmoveto":
                raise Exception("first operator is not rmoveto")
            
            im_pixel_size = (128, 128)
            crop_factor = 1
            dpi = 1
            boundaries = (int((im_pixel_size[0] * (crop_factor * 100 / dpi - 1)) // 2), int((im_pixel_size[1] * (crop_factor * 100 / dpi - 1)) // 2))
            im_size_inches = ((im_pixel_size[0] * crop_factor) / dpi, (im_pixel_size[1] * crop_factor) / dpi)
            img_arr = viz.draw(
                display=False,
                filename=f"./fontmakerai/training_images/{epoch+1}_{batch_idx+1}.png",
                return_image=True,
                center=False,
                im_size_inches=im_size_inches,
                bounds=(-300, 300),
                dpi=dpi
            )[None,:,:,0]
            
            im_cpu = (im[0] * 127.5 + 127.5).to(device=device, dtype=torch.uint8).cpu().detach().numpy()
            img_arr = wandb.Image(np.concatenate([im_cpu, img_arr], axis=2), caption=f"epoch{epoch+1}_{batch_idx+1}.png")
        except Exception as e:
            flag = False
            print(f"Could not generate visualization; generated output was not formatted correctly: {e}")
    model.train()
    
    if flag:
        return (wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}_{batch_idx+1}.png"), img_arr)
    else:
        return (wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}_{batch_idx+1}.png"), None)


# ### Pretrain Vision Transformer Encoder

# In[ ]:


if args["pretrain_vit_encoder"] and not args["use_pretrained_vit_encoder"]:
    print("\nPretraining ViT encoder...\n")

    model.train()
    for epoch in range(args['pretrain_vit_encoder_epochs']):
        total_loss = 0
        last_loss = 0
        pretrain_vit_encoder_optimizer.zero_grad()
        for idx, (X, im) in enumerate(tqdm(cff_pretrain_vit_encoder_dataloader)):
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            out = model.encoder.pretrain(im)
            loss = recon_loss(out, im) / X.shape[0]
            total_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.cuda.empty_cache()

            if (idx+1) % args['pretrain_vit_encoder_batch_accumulate'] == 0:
                pretrain_vit_encoder_optimizer.step()
                pretrain_vit_encoder_optimizer.zero_grad()
                if args['pretrain_vit_encoder_use_scheduler']:
                    pretrain_vit_encoder_scheduler.step()
                diff = total_loss - last_loss
                last_loss = total_loss

            if args['use_wandb']:
                if (idx+1) % args['pretrain_vit_encoder_batch_accumulate'] == 0:
                    wandb.log({
                        "pretrain_vit_encoder_loss_step": diff / (args['pretrain_vit_encoder_batch_accumulate'] * args['pretrain_vit_encoder_batch_size']),
                        "pretrain_vit_encoder_lr_step": pretrain_vit_encoder_scheduler.get_last_lr()[0] if args['pretrain_vit_encoder_use_scheduler'] else args['pretrain_vit_encoder_lr'],
                        "original_image": wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png"),
                        "reconstructed_image": wandb.Image(out[0].clamp(-1, 1).to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png")
                    })
        print(f"Epoch {epoch+1}/{args['pretrain_vit_encoder_epochs']} completed. Total Loss = {total_loss/cff_dataset.shape[0]}")
        pretrain_vit_encoder_optimizer.zero_grad()
        torch.cuda.empty_cache()
    torch.save(model.encoder, f'models/pretrained_vit_encoder-{args["embedding_dim"]}.pt')
elif args["use_pretrained_vit_encoder"]:
    print("\nUsing pretrained ViT encoder...\n")
    model.encoder = torch.load(f'models/pretrained_vit_encoder-{args["embedding_dim"]}.pt')


# ### Pretrain Embeddings

# In[ ]:


if args["pretrain_embeddings"] and not args["use_pretrained_embeddings"]:
    print("\nPretraining embeddings...\n")

    model.train()
    for epoch in range(args['pretrain_epochs']):
        total_loss = 0
        for (X, im) in tqdm(cff_pretrain_dataloader):
            pretrain_optimizer.zero_grad()
            inputs = X.to(device, dtype=torch.int32)
            out = model.identity_embeddings(inputs)
            if epoch <= 0:
                loss = numeric_mse_loss(out, inputs) / inputs.shape[0]
            else:
                loss = test_loss_fn(out.permute(0, 2, 1), inputs.long()) / inputs.shape[0]
            total_loss += loss.item() * inputs.shape[0]
            loss.backward()
            pretrain_optimizer.step()
            torch.cuda.empty_cache()

            if args['pretrain_use_scheduler']:
                pretrain_scheduler.step()

            if args['use_wandb']:
                wandb.log({
                    "pretrain_loss_step": loss.item() / inputs.shape[0],
                    "pretrain_lr_step": pretrain_scheduler.get_last_lr()[0] if args['pretrain_use_scheduler'] else args['pretrain_lr']
                })
        print(f"Epoch {epoch+1}/{args['pretrain_epochs']} completed. Total Loss = {total_loss/cff_dataset.shape[0]}")

    torch.save(model.decoder.embedder.weight, f'models/pretrained_embeddings-{args["embedding_dim"]}-numeric.pt')
    torch.save(model.decoder.inverse_embedder.weight, f'models/pretrained_decoder_inverse_embedder-{args["embedding_dim"]}-numeric.pt')
    torch.save(model.decoder.command_encoder.weight, f'models/pretrained_command_encoder-{args["embedding_dim"]}-numeric.pt')
    torch.save(model.decoder.command_decoder.weight, f'models/pretrained_command_decoder-{args["embedding_dim"]}-numeric.pt')
    torch.save(model.decoder.norm_final.weight, f'models/pretrained_norm_final-{args["embedding_dim"]}-numeric.pt')
elif args["use_pretrained_embeddings"]:
    print("\nUsing pretrained embeddings...\n")
    model.decoder.embedder.weight = torch.load(f'models/pretrained_embeddings-{args["embedding_dim"]}-numeric.pt', weights_only=True)
    model.decoder.inverse_embedder.weight = torch.load(f'models/pretrained_decoder_inverse_embedder-{args["embedding_dim"]}-numeric.pt', weights_only=True)
    model.decoder.command_encoder.weight = torch.load(f'models/pretrained_command_encoder-{args["embedding_dim"]}-numeric.pt', weights_only=True)
    model.decoder.command_decoder.weight = torch.load(f'models/pretrained_command_decoder-{args["embedding_dim"]}-numeric.pt', weights_only=True)
    model.decoder.norm_final.weight = torch.load(f'models/pretrained_norm_final-{args["embedding_dim"]}-numeric.pt', weights_only=True)

if args['freeze_embeddings']:
    print("\nFreezing embeddings...\n")
    model.decoder.embedder.weight.requires_grad = False
    model.decoder.inverse_embedder.weight.requires_grad = False
    model.decoder.command_encoder.weight.requires_grad = False
    model.decoder.command_decoder.weight.requires_grad = False
    model.decoder.norm_final.weight.requires_grad = False


# ### Train Transformer

# In[ ]:


print("\nTraining model...\n")

if args['train_transformer']:
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    src = torch.zeros((args['batch_size'], 0)).to(device)
    for epoch in range(args['epochs']):
        model.train()
        optimizer.zero_grad()
        total_loss = 0
        last_loss = 0
        train_batches = int((max_len*(num_glyphs // step_every)*0.95) // args['batch_size']) + 1
        # train_batches = 1000
        for idx, (X, im) in enumerate(tqdm(cff_train_dataloader, total=train_batches)):
            if idx >= train_batches:
                break
            inputs = X.to(device, dtype=torch.int32)
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            out = model(im, inputs[:,:-7]) # Use only output tokens before this truth term
            
            # loss = loss_fn(out.permute(0, 2, 1), inputs.long()) / X.shape[0]
            loss = numeric_mse_loss(out, inputs) / X.shape[0]

            total_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.cuda.empty_cache()

            if (idx+1) % args['batch_accumulate'] == 0 or idx == train_batches-1:
                if args['gradient_clip']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args['gradient_clip_val'])
                optimizer.step()
                optimizer.zero_grad()
                if args['use_scheduler']:
                    scheduler.step()
                diff = total_loss - last_loss
                last_loss = total_loss

            if args['use_wandb']:
                if (idx+1) % (100 * args['batch_accumulate']) == 0 or (idx == train_batches-1 and (epoch+1) % args['sample_every'] == 0):
                    goal_image, img_arr = decode(epoch, idx)
                    wandb.log({
                        "goal_image": goal_image,
                        "images": img_arr,
                        "train_loss_step": diff / (args['batch_accumulate'] * args['batch_size']),
                        "lr_step": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                    })
                elif (idx+1) % args['batch_accumulate'] == 0:
                    wandb.log({
                        "train_loss_step": diff / (args['batch_accumulate'] * args['batch_size']),
                        "lr_step": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                    })
        # train_loss_list += [total_loss / cff_dataset.shape[0]]
        train_loss_list += [total_loss / (min(train_batches, idx+1)*args['batch_size'])]
        
        model.eval()
        total_loss = 0
        test_batches = 25
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        with torch.no_grad():
            for idx, (X, im) in enumerate(tqdm(cff_test_dataloader, total=test_batches)):
                if idx >= test_batches:
                    break
                inputs = X.to(device, dtype=torch.int32)
                im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
                out = model(im, inputs[:,:-7]) # Use only output tokens before this truth term

                # loss = loss_fn(out.permute(0, 2, 1), inputs.long()) / X.shape[0]
                loss = numeric_mse_loss(out, inputs) / X.shape[0]
                
                total_loss += loss.item() * X.shape[0]
                torch.cuda.empty_cache()

                guesses = out.permute(0, 2, 1).argmax(dim=1)
                truths = inputs
                true_positives += ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
                false_positives += ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
                true_negatives += ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
                false_negatives += ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
            
            # test_loss_list += [total_loss / cff_dataset_test.shape[0]]
            test_loss_list += [total_loss / (min(test_batches, idx+1)*args['batch_size'])]
            acc, pre, rec, f1 = PerformanceMetrics.all_metrics(
                tp=true_positives,
                fp=false_positives,
                tn=true_negatives,
                fn=false_negatives
            )

            print(f"Epoch {epoch+1}/{args['epochs']} completed. Train Loss = {train_loss_list[-1]};  Test Loss: {test_loss_list[-1]}")

            if args['use_wandb']:
                wandb.log({
                    "train_loss": train_loss_list[-1],
                    "test_loss": test_loss_list[-1],
                    # "test_accuracy": acc,
                    # "test_precision": pre,
                    # "test_recall": rec,
                    # "test_f1": f1,
                    "lr": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                })

        # if (epoch+1) % 100 == 0 or epoch+1 == args['epochs']:
        if max_len > 100:
            torch.save(model, f'models/transformer-{dataset_name}-{epoch+1}.pkl')


# ### Reinforcement Learning

# In[ ]:


print("\nPost-training model...\n")

if args['post_train']:
    # Base LLM
    # Now, `model` is the fine-tuned policy model
    original_model = deepcopy(model)
    original_model.eval()

    @torch.no_grad()
    def ground_truth_reward_nums(ground_truth_num_tokens):
        '''
        ground_truth_tokens: (batch_size, seq_len)
        '''
        vocab_tensor = torch.arange(0, tokenizer.num_tokens, dtype=torch.int32).unsqueeze(0).unsqueeze(0).to(device)
        reward = (-(vocab_tensor - ground_truth_num_tokens.unsqueeze(-1)).abs()).exp() * 10
        reward[:,:,1:35] = -10
        reward = torch.where(ground_truth_num_tokens.unsqueeze(-1).repeat(1, 1, reward.shape[2]) == 0, torch.zeros_like(reward), reward)
        return reward

    @torch.no_grad()
    def ground_truth_reward_ops(ground_truth_op_tokens):
        '''
        ground_truth_tokens: (batch_size, seq_len)
        '''
        vocab_tensor = torch.arange(0, tokenizer.num_tokens, dtype=torch.int32).unsqueeze(0).unsqueeze(0).to(device)
        reward = ((vocab_tensor == ground_truth_op_tokens.unsqueeze(-1)) * 10)
        reward[:,:,35:] = -10
        reward = torch.where(ground_truth_op_tokens.unsqueeze(-1).repeat(1, 1, reward.shape[2]) == 0, torch.zeros_like(reward), reward)
        return reward

    # Define reward function
    def reward_fn(ground_truth_tokens, original_policy_logits, new_policy_logits):
        # Calculate the loss between the output and the input
        pred_tokens = new_policy_logits.argmax(dim=-1)
        pred_op_tokens = pred_tokens[:,::7] # (batch_size, seq_len / 7)
        pred_num_tokens = pred_tokens.view(pred_tokens.shape[0], pred_tokens.shape[1]//7, 7)[:,:,1:].flatten(start_dim=1) # (batch_size, seq_len * 6 / 7)
        truth_op_tokens = ground_truth_tokens[:,::7] # (batch_size, seq_len / 7)
        truth_num_tokens = ground_truth_tokens.view(ground_truth_tokens.shape[0], ground_truth_tokens.shape[1]//7, 7)[:,:,1:].flatten(start_dim=1) # (batch_size, seq_len * 6 / 7)
        bs = ground_truth_tokens.shape[0]
        seq_len = ground_truth_tokens.shape[1]
        num_logits = new_policy_logits.shape[2]
        op_reward = ground_truth_reward_ops(truth_op_tokens)
        num_reward = ground_truth_reward_nums(truth_num_tokens)
        gt_rewards = torch.cat([op_reward.view((bs, seq_len//7, 1, num_logits)), num_reward.view((bs, seq_len//7, 6, num_logits))], dim=2).flatten(start_dim=1, end_dim=2) # (batch_size, seq_len, num_logits)
        reward = (gt_rewards * new_policy_logits.softmax(dim=-1)) # (batch_size, seq_len, num_logits)
        kl_penalty = log_kl_loss(new_policy_logits.log_softmax(dim=-1), original_policy_logits.log_softmax(dim=-1))
        # Return the negative loss as the reward
        return -reward.sum() + kl_penalty * args['post_train_kl_penalty']
    
    src = torch.zeros((args['post_train_batch_size'], 0)).to(device)
    for epoch in range(args['post_train_epochs']):
        model.train()
        posttrain_optimizer.zero_grad()
        total_loss = 0
        last_loss = 0
        train_batches = (max_len*(num_glyphs // step_every) // args['post_train_batch_size']) + 1
        for idx, (X, im) in enumerate(tqdm(cff_posttrain_dataloader, total=train_batches)):
            if idx >= train_batches:
                break
            inputs = X.to(device, dtype=torch.int32)
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            out_new = model.decode(im, None, decode_instr)[0].cpu().detach().numpy().flatten()
            with torch.no_grad():
                out_original = original_model(im, out_new[:,:-7]) # Use only output tokens before this truth term
            
            loss = reward_fn(inputs, out_original, out_new) / X.shape[0]

            total_loss += loss.item() * X.shape[0]
            loss.backward()
            torch.cuda.empty_cache()

            if (idx+1) % 1 == 0 or idx == train_batches-1:
                if args['gradient_clip']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args['gradient_clip_val'])
                posttrain_optimizer.step()
                posttrain_optimizer.zero_grad()
                if args['post_train_use_scheduler']:
                    posttrain_scheduler.step()
                diff = total_loss - last_loss
                last_loss = total_loss

            if args['use_wandb']:
                if (idx+1) % 100 == 0 or (idx == train_batches-1 and (epoch+1) % args['sample_every'] == 0):
                    goal_image, img_arr = decode(epoch, idx)
                    wandb.log({
                        "posttrain_goal_image": goal_image,
                        "posttrain_images": img_arr,
                        "posttrain_loss_step": diff / (args['batch_accumulate'] * args['post_train_batch_size']),
                        "posttrain_lr_step": args['post_train_lr'] if not args['post_train_use_scheduler'] else posttrain_scheduler.get_last_lr()[0],
                    })
                elif (idx+1) % 1 == 0:
                    wandb.log({
                        "posttrain_loss_step": diff / (args['batch_accumulate'] * args['post_train_batch_size']),
                        "posttrain_lr_step": args['post_train_lr'] if not args['post_train_use_scheduler'] else posttrain_scheduler.get_last_lr()[0],
                    })
        train_loss = total_loss / (min(train_batches, idx+1)*args['batch_size'])
        
        model.eval()
        total_loss = 0
        test_batches = 25
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        with torch.no_grad():
            for idx, (X, im) in enumerate(tqdm(cff_test_dataloader, total=test_batches)):
                if idx >= test_batches:
                    break
                inputs = X.to(device, dtype=torch.int32)
                im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
                out = model(im, inputs[:,:-7]) # Use only output tokens before this truth term

                # loss = loss_fn(out.permute(0, 2, 1), inputs.long()) / X.shape[0]
                loss = numeric_mse_loss(out, inputs) / X.shape[0]
                
                total_loss += loss.item() * X.shape[0]
                torch.cuda.empty_cache()

                guesses = out.permute(0, 2, 1).argmax(dim=1)
                truths = inputs
                true_positives += ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
                false_positives += ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
                true_negatives += ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
                false_negatives += ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
            
            test_loss = total_loss / (min(test_batches, idx+1)*args['batch_size'])
            acc, pre, rec, f1 = PerformanceMetrics.all_metrics(
                tp=true_positives,
                fp=false_positives,
                tn=true_negatives,
                fn=false_negatives
            )

            print(f"Epoch {epoch+1}/{args['epochs']} completed. Train Loss = {train_loss_list[-1]};  Test Loss: {test_loss_list[-1]}")

            if args['use_wandb']:
                wandb.log({
                    "posttrain_loss": train_loss,
                    "posttrain_test_loss": test_loss,
                    # "test_accuracy": acc,
                    # "test_precision": pre,
                    # "test_recall": rec,
                    # "test_f1": f1,
                    "lr": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                })

        # if (epoch+1) % 100 == 0 or epoch+1 == args['epochs']:
        if max_len > 100:
            torch.save(model, f'models/transformer-{dataset_name}-{epoch+1}.pkl')

