#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from config import conv_map, device, operators

from fontmodel import (FontModel, DecodeInstruction, DecodeType, SamplingType)
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from performance import PerformanceMetrics
from tablelist_utils import numbers_first, make_non_cumulative


# ### Training arguments

# In[ ]:


print(f"Executing train-cff.ipynb on {device}...\n-----------------------------")

args = {
    "load_model": False,
    "train_transformer": True,
    "min_number": -500,
    "max_number": 500,
    "max_seq_len": 5040,
    "num_layers": 6,
    "embedding_dim": 512,
    "num_heads": 16,
    "ff_dim": 768 * 2,
    "use_wandb": True,
    "epochs": 25,
    "batch_size": 64,
    "lr": 1e-4,
    "dropout_rate": 0.1,
    "weight_decay": 1e-1,
    "gradient_clip": True,
    "gradient_clip_val": 1.0,
    "label_smoothing": 0.001,
    "sample_every": 1,
    "use_scheduler": True,
    "scheduler_warmup_steps": 5000,
    "data_type": torch.bfloat16,
    "vae_beta": 1e-1,
    "vae_epochs": 10,
    "vae_lr": 1e-2,
    "vae_weight_decay": 1e-5,
    "freeze_embeddings": False,
    "use_pretrained_embeddings": False,
    "pretrain_embeddings": True,
    "pretrain_epochs": 2,
    "pretrain_batch_size": 128,
    "pretrain_lr": 4e-3,
    "pretrain_use_scheduler": True,
    "pretrain_scheduler_warmup_steps": 3000,
    "use_pretrained_vit_encoder": False,
    "pretrain_vit_encoder": True,
    "pretrain_vit_encoder_epochs": 1,
    "pretrain_vit_encoder_batch_size": 128,
    "pretrain_vit_encoder_lr": 1e-2,
    "pretrain_vit_encoder_weight_decay": 1e-1,
    "pretrain_vit_encoder_use_scheduler": True,
    "pretrain_vit_encoder_scheduler_warmup_steps": 50
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
    model = torch.load(f'models/ldm-basic-35851allchars-0.pkl', map_location=device).to(device)
else:
    model = FontModel(
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
v = count_params(model.embedder)


# ### Training tools

# In[ ]:


# Parameters (tentative):
# FontModel: embedder (DON'T APPLY WEIGHT DECAY)
# TransformerDecoder: transformer_decoder_layers (DON'T APPLY WEIGHT DECAY TO RMSNORM), command_encoder, command_decoder, norm_final (DON'T APPLY WEIGHT DECAY)
# TransformerEncoder: transformer_encoder_layers (DON'T APPLY WEIGHT DECAY TO RMSNORM), embedder (custom),pos_embed, norm_final (DON'T APPLY WEIGHT DECAY)

# We don't want to apply weight decay to layer norms and embeddings
no_weight_decay_params = [x for x in model.embedder.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.decoder.transformer_decoder_layers.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
no_weight_decay_params += [x for x in model.decoder.norm_final.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
no_weight_decay_params += [x for x in model.encoder.norm_final.parameters() if x.requires_grad]
no_weight_decay_params += [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]

weight_decay_params = [x for name, x in model.decoder.transformer_decoder_layers.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for x in model.decoder.command_encoder.parameters() if x.requires_grad]
weight_decay_params += [x for x in model.decoder.command_decoder.parameters() if x.requires_grad]
weight_decay_params += [x for name, x in model.encoder.transformer_encoder_layers.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
weight_decay_params += [x for x in model.encoder.pos_embed.parameters() if x.requires_grad]

vit_encoder_params_nwd = [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
vit_encoder_params_nwd += [x for name, x in model.encoder.pretrain_reverse_ae.named_parameters() if x.requires_grad and ('norm' in name or 'bias' in name)]
vit_encoder_params_nwd += [x for x in model.encoder.norm_final.parameters() if x.requires_grad]
vit_encoder_params_wd = [x for name, x in model.encoder.embedder.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]
vit_encoder_params_wd += [x for name, x in model.encoder.pretrain_reverse_ae.named_parameters() if x.requires_grad and 'norm' not in name and 'bias' not in name]

optimizer = torch.optim.AdamW(
    [
       {'params': weight_decay_params, 'weight_decay': args['weight_decay']},
       {'params': no_weight_decay_params, 'weight_decay': 0.0}
    ],
    betas=(0.9, 0.95),
    lr=args['lr']
)

pretrain_optimizer = torch.optim.AdamW(no_weight_decay_params, weight_decay=0.0, betas=(0.9, 0.95), lr=args['pretrain_lr'])

pretrain_vit_encoder_optimizer = torch.optim.AdamW(
    [
        {'params': vit_encoder_params_wd, 'weight_decay': args['pretrain_vit_encoder_weight_decay']},
        {'params': vit_encoder_params_nwd, 'weight_decay': 0.0},
    ],
    betas=(0.9, 0.95),
    lr=args['pretrain_vit_encoder_lr']
)

if args['use_scheduler']:
    # scheduler = TransformerScheduler(
    #     optimizer=optimizer,
    #     dim_embed=args['embedding_dim'],
    #     warmup_steps=args['scheduler_warmup_steps']
    # )
    batches_per_epoch = int(33928 * 26 / args['batch_size'] + 0.5)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * batches_per_epoch, eta_min=1e-5)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['scheduler_warmup_steps'])
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)

    pretrain_batches_per_epoch = int(33928 * 26 / args['pretrain_batch_size'] + 0.5)
    pretrain_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=args['pretrain_epochs'] * pretrain_batches_per_epoch, eta_min=1e-5)
    pretrain_scheduler2 = torch.optim.lr_scheduler.LinearLR(pretrain_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['pretrain_scheduler_warmup_steps'])
    pretrain_scheduler = torch.optim.lr_scheduler.ChainedScheduler([pretrain_scheduler1, pretrain_scheduler2], optimizer=pretrain_optimizer)

    pretrain_vit_encoder_batches_per_epoch = int(33928 * 26 / args['pretrain_vit_encoder_batch_size'] + 0.5)
    pretrain_vit_encoder_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_vit_encoder_optimizer, T_max=args['pretrain_vit_encoder_epochs'] * pretrain_vit_encoder_batches_per_epoch, eta_min=1e-5)
    pretrain_vit_encoder_scheduler2 = torch.optim.lr_scheduler.LinearLR(pretrain_vit_encoder_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['pretrain_vit_encoder_scheduler_warmup_steps'])
    pretrain_vit_encoder_scheduler = torch.optim.lr_scheduler.ChainedScheduler([pretrain_vit_encoder_scheduler1, pretrain_vit_encoder_scheduler2], optimizer=pretrain_vit_encoder_optimizer)

dataset_name = "basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded"
max_len = 33928
num_glyphs = 26
step_every = 1
train_start, train_end = 0, int(0.95 * max_len) * num_glyphs
test_start, test_end = train_end, max_len * num_glyphs
# train_start, train_end = 0, 26*1
# test_start, test_end = 0, 26*1
cff_dataset = torch.load(f'./{dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
cff_dataset_test = torch.load(f'./{dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
im_dataset_name = "basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)"
im_dataset = torch.load(f'./{im_dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
im_dataset_test = torch.load(f'./{im_dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
cff_train_tensor_dataset = TensorDataset(cff_dataset, im_dataset)
cff_train_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['batch_size'], shuffle=True)
cff_pretrain_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['pretrain_batch_size'], shuffle=True)
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

def kl_loss_fn(mu, logvar):
    return 0.5 * ((torch.pow(mu, 2) + logvar.exp() - logvar - 1)).sum()

curve_width = 7
first_numeric_idx = 3 + len(tokenizer.possible_operators)
zero_mask = (torch.ones((1,1,1,vocab_size)) * (torch.arange(0,vocab_size, dtype=torch.int32) >= first_numeric_idx)).to(device)
hlf = curve_width // 2
offset = torch.arange(-hlf, hlf+1, dtype=torch.int32)[None,None,:].to(device) # (1, 1, curve_width)
neg_offset_exp = (-offset.abs().unsqueeze(-1)).exp()
kl_loss = torch.nn.KLDivLoss(reduction='sum')
cross_entropy_loss = torch.nn.CrossEntropyLoss(
    reduction='sum',
    ignore_index=tokenizer[tokenizer.pad_token],
    label_smoothing=args['label_smoothing']
)
bce_loss = nn.BCELoss()

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
        # single_tgt = torch.nn.functional.one_hot(targets.long(), tokenizer.num_tokens).int() # (batch_size, seq_len, vocab_size)
        
        token_count = targets.shape[0] * targets.shape[1] * curve_width
        arrng = torch.arange(0, token_count, dtype=torch.int32)
        batch_indices = torch.floor_divide(arrng, targets.shape[1] * curve_width).unsqueeze(0).to(device)
        sequence_indices = torch.floor_divide(torch.remainder(arrng, targets.shape[1] * curve_width), curve_width).unsqueeze(0).to(device)
        curve_indices = torch.remainder(arrng, curve_width).unsqueeze(0).to(device)
        token_indices = torch.clip(targets.unsqueeze(-1) + offset, min=1, max=tokenizer.num_tokens-1).flatten().unsqueeze(0)
        multi_tgt = torch.sparse_coo_tensor(
            indices=torch.cat([batch_indices, sequence_indices, curve_indices, token_indices], dim=0),
            values=torch.ones(token_count,).to(device),
            size=(targets.shape[0], targets.shape[1], curve_width, tokenizer.num_tokens)
        ) # (batch_size, seq_len, curve_width, vocab_size)
        multi_tgt = multi_tgt * neg_offset_exp * zero_mask
        # multi_tgt = torch.clip(targets.unsqueeze(-1) + offset, min=1, max=tokenizer.num_tokens-1) # (batch_size, seq_len, curve_width)
        # multi_tgt = torch.nn.functional.one_hot(multi_tgt.long(), tokenizer.num_tokens).int() * neg_offset_exp # (batch_size, seq_len, curve_width, vocab_size)
        # multi_tgt[:,:,:,:first_numeric_idx] = 0
        multi_tgt = (multi_tgt.sum(dim=-2).to_dense() / (1e-10 + multi_tgt.sum(dim=(-2,-1)).unsqueeze(-1).to_dense())).to_sparse() # (batch_size, seq_len, vocab_size)
        tgt_dist = is_numeric_tgt * multi_tgt + (~is_numeric_tgt) * (is_non_padding) * single_tgt # (batch_size, seq_len, vocab_size)

    return kl_loss(output.log_softmax(dim=-1), tgt_dist.to_dense())# + cross_entropy_loss(output, targets)


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
            toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

            print("Before:", toks)
            with open(f"./fontmakerai/training_images/{epoch+1}_{batch_idx+1}.txt", 'w') as f:
                j_str = '\', \''
                f.write(f"Before: ['{j_str.join([str(x) for x in toks])}']\n\n")
                f.write(f"decode_instr:")
                for k, v in decode_instr.__dict__.items():
                    f.write(f"\n\t{k}={v}")
                f.write("\n")
            toks = [tok for tok in toks if tok != '<PAD2>']
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
            
            im_pixel_size = (128, 128)
            crop_factor = 1.5
            boundaries = (int((im_pixel_size[0] * (crop_factor - 1)) // 2), int((im_pixel_size[1] * (crop_factor - 1)) // 2))
            ppi = 100
            im_size_inches = ((im_pixel_size[0] * crop_factor) / ppi, (im_pixel_size[1] * crop_factor) / ppi)
            img_arr = viz.draw(
                display=False,
                filename=f"./fontmakerai/training_images/{epoch+1}_{batch_idx+1}.png",
                return_image=True,
                center=True
            )[None,:,:,0]
            
            img_arr = wandb.Image(img_arr, caption=f"epoch{epoch+1}_{batch_idx+1}.png")
        except Exception as e:
            flag = False
            print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")
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
        for (X, im) in tqdm(cff_pretrain_dataloader):
            pretrain_vit_encoder_optimizer.zero_grad()
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            out = model.encoder.pretrain(im)
            loss = recon_loss(out, im) / X.shape[0]
            # loss = bce_loss(out / 2.0 + 0.5, im / 2.0 + 0.5) / X.shape[0]
            total_loss += loss.item() * X.shape[0]
            loss.backward()
            pretrain_vit_encoder_optimizer.step()
            torch.cuda.empty_cache()

            if args['pretrain_vit_encoder_use_scheduler']:
                pretrain_vit_encoder_scheduler.step()

            if args['use_wandb']:
                wandb.log({
                    "pretrain_vit_encoder_loss_step": loss.item() / X.shape[0],
                    "pretrain_vit_encoder_lr_step": pretrain_vit_encoder_scheduler.get_last_lr()[0] if args['pretrain_vit_encoder_use_scheduler'] else args['pretrain_vit_encoder_lr'],
                    "original_image": wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png"),
                    "reconstructed_image": wandb.Image(out[0].clamp(-1, 1).to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png")
                })
        print(f"Epoch {epoch+1}/{args['pretrain_vit_encoder_epochs']} completed. Total Loss = {total_loss/cff_dataset.shape[0]}")

    torch.save(model.encoder.embedder, f'models/pretrained_vit_encoder-{args["embedding_dim"]}.pt')
elif args["use_pretrained_vit_encoder"]:
    print("\nUsing pretrained ViT encoder...\n")
    model.encoder.embedder = torch.load(f'models/pretrained_vit_encoder-{args["embedding_dim"]}.pt')


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

    torch.save(model.embedder.weight, f'models/pretrained_embeddings-{args["embedding_dim"]}.pt')
    torch.save(model.decoder.command_encoder.weight, f'models/pretrained_command_encoder-{args["embedding_dim"]}.pt')
    torch.save(model.decoder.command_decoder.weight, f'models/pretrained_command_decoder-{args["embedding_dim"]}.pt')
    torch.save(model.decoder.norm_final.weight, f'models/pretrained_norm_final-{args["embedding_dim"]}.pt')
elif args["use_pretrained_embeddings"]:
    print("\nUsing pretrained embeddings...\n")
    model.embedder.weight = torch.load(f'models/pretrained_embeddings-{args["embedding_dim"]}.pt')
    model.decoder.command_encoder.weight = torch.load(f'models/pretrained_command_encoder-{args["embedding_dim"]}.pt')
    model.decoder.command_decoder.weight = torch.load(f'models/pretrained_command_decoder-{args["embedding_dim"]}.pt')
    model.decoder.norm_final.weight = torch.load(f'models/pretrained_norm_final-{args["embedding_dim"]}.pt')

if args['freeze_embeddings']:
    print("\nFreezing embeddings...\n")
    model.embedder.weight.requires_grad = False
    model.decoder.command_encoder.weight.requires_grad = False
    model.decoder.command_decoder.weight.requires_grad = False
    model.decoder.norm_final.weight.requires_grad = False


# ### Train VAE for operator compression

# In[ ]:


# print("\nTraining model for operator compression...\n")

# # Note: in training, padding is included in the loss. This can be removed later during inference.
# if args['train_transformer']:
#     op_vae_optimizer = torch.optim.AdamW(op_vae.parameters(), lr=args['vae_lr'], weight_decay=args['vae_weight_decay'])
#     for epoch in range(args['vae_epochs']):
#         op_vae.train()
#         total_loss = 0
#         for (X, _) in tqdm(cff_train_dataloader):
#             inputs = X.to(device)
            
#             op_vae_optimizer.zero_grad()
#             inp, inp_hat, mu, logvar = op_vae(inputs, tokenizer)
#             loss = (numeric_mse_loss(inp_hat, inp) + args['vae_beta'] * kl_loss_fn(mu, logvar)) / inp.shape[0]
#             total_loss += loss.item()
#             loss.backward()
#             op_vae_optimizer.step()
#             torch.cuda.empty_cache()

#         op_vae.eval()
#         test_loss = 0
#         with torch.no_grad():
#             for (X, _) in tqdm(cff_test_dataloader):
#                 inputs = X.to(device)
                
#                 inp, inp_hat, mu, logvar = op_vae(inputs, tokenizer)
#                 loss = (numeric_mse_loss(inp_hat, inp) + args['vae_beta'] * kl_loss_fn(mu, logvar)) / inp.shape[0]
#                 test_loss += loss.item()

#             test_val = torch.IntTensor([[9, 302, 305, 122, 955, 1002, 554, 36]]).to(device)
#             inp, inp_hat, mu, logvar = op_vae(test_val, tokenizer)
#             loss = (numeric_mse_loss(inp_hat, inp) + args['vae_beta'] * kl_loss_fn(mu, logvar))
#             print(f"loss: {loss}, mu: {mu.abs().mean().item()}, logvar: {logvar.abs().mean().item()}")
#             print(inp_hat.argmax(dim=1).cpu().detach().numpy().flatten().tolist())

#             if args['use_wandb']:
#                 wandb.log({
#                     "train_loss": total_loss / cff_dataset.shape[0],
#                     "test_loss": test_loss / cff_dataset_test.shape[0]
#                 })
#         print(f"epoch {epoch+1}/{args['vae_epochs']}: train loss: {total_loss / cff_dataset.shape[0]}     test loss: {test_loss / cff_dataset_test.shape[0]}")

#     torch.save(op_vae, f'models/op_vae-{dataset_name}.pkl')


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
        total_loss = 0
        train_batches = (max_len*num_glyphs // args['batch_size']) + 1
        for idx, (X, im) in enumerate(tqdm(cff_train_dataloader, total=train_batches)):
            if idx >= train_batches:
                break
            inputs = X.to(device, dtype=torch.int32)
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            optimizer.zero_grad()
            out = model(im, inputs[:,:-7]) # Use only output tokens before this truth term
            loss = loss_fn(out.permute(0, 2, 1), inputs.long())
            # loss = numeric_mse_loss(out, inputs)
            total_loss += loss.item()
            loss.backward()
            if args['gradient_clip']:
                torch.nn.utils.clip_grad_value_(model.parameters(), args['gradient_clip_val'])
            optimizer.step()
            if args['use_scheduler']:
                scheduler.step()
            torch.cuda.empty_cache()

            if args['use_wandb']:
                if (idx+1) % 250 == 0:
                    goal_image, img_arr = decode(epoch, idx)
                    wandb.log({
                        "goal_image": goal_image,
                        "images": img_arr,
                        "train_loss_step": loss.item() / inputs.shape[0],
                        "lr_step": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                    })
                else:
                    wandb.log({
                        "train_loss_step": loss.item() / inputs.shape[0],
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
                loss = loss_fn(out.permute(0, 2, 1), inputs.long())
                # loss = numeric_mse_loss(out, inputs)
                total_loss += loss.item()
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
            # torch.save(model, './fontmakerai/model.pkl')
            
            if (epoch + 1) % args['sample_every'] == 0:
                with open('./fontmakerai/.config.txt', 'r') as cf:
                    lines = cf.readlines()
                    if len(lines) != 7:
                        print(f"Not decoding this iteration; .config.txt has wrong number of lines ({len(lines)})")
                        continue
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

                try:
                    flag = True
                    idx = np.random.randint(0, im_dataset_test.shape[0])
                    im = im_dataset_test[idx:idx+1].to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
                    sequence = model.decode(im, None, decode_instr)[0].cpu().detach().numpy().flatten()
                    torch.cuda.empty_cache()
                    # sequence = cff_train_tensor_dataset[0:1][0][0].cpu().detach().numpy().flatten()#.to(device)
                    toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

                    print("Before:", toks)
                    with open(f"./fontmakerai/training_images/{epoch+1}.txt", 'w') as f:
                        j_str = '\', \''
                        f.write(f"Before: ['{j_str.join([str(x) for x in toks])}']\n\n")
                        f.write(f"decode_instr:")
                        for k, v in decode_instr.__dict__.items():
                            f.write(f"\n\t{k}={v}")
                        f.write("\n")
                    toks = [tok for tok in toks if tok != '<PAD2>']
                    if cumulative:
                        toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
                        # toks = make_non_cumulative(toks, tokenizer)
                    else:
                        toks = numbers_first(toks, tokenizer, return_string=False)
                        # toks = toks
                    print("After:", toks)
                    viz = Visualizer(toks)
                    with open(f"./fontmakerai/training_images/{epoch+1}.txt", 'a', newline='\n') as f:
                        j_str = '\', \''
                        f.write(f"After: ['{j_str.join([str(x) for x in toks])}']")
                    
                    im_pixel_size = (128, 128)
                    crop_factor = 1.5
                    boundaries = (int((im_pixel_size[0] * (crop_factor - 1)) // 2), int((im_pixel_size[1] * (crop_factor - 1)) // 2))
                    ppi = 100
                    im_size_inches = ((im_pixel_size[0] * crop_factor) / ppi, (im_pixel_size[1] * crop_factor) / ppi)
                    img_arr = viz.draw(
                        display=False,
                        filename=f"./fontmakerai/training_images/{epoch+1}.png",
                        return_image=True,
                        center=True
                    )[None,:,:,0]
                    
                    img_arr = wandb.Image(img_arr, caption=f"epoch{epoch+1}.png")
                    # wandb.log({"images": img_arr}) # TODO: also log decoder instructions
                except Exception as e:
                    flag = False
                    print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")

            
            if (epoch + 1) % args['sample_every'] == 0 and flag:
                if args['use_wandb']:
                    wandb.log({
                        "train_loss": train_loss_list[-1],
                        "test_loss": test_loss_list[-1],
                        "test_accuracy": acc,
                        "test_precision": pre,
                        "test_recall": rec,
                        "test_f1": f1,
                        "lr": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                        "goal_image": wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png"),
                        "images": img_arr
                    })
            elif (epoch + 1) % args['sample_every'] == 0:
                if args['use_wandb']:
                    wandb.log({
                        "train_loss": train_loss_list[-1],
                        "test_loss": test_loss_list[-1],
                        "test_accuracy": acc,
                        "test_precision": pre,
                        "test_recall": rec,
                        "test_f1": f1,
                        "lr": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                        "goal_image": wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}.png"),
                    })
            else:
                if args['use_wandb']:
                    wandb.log({
                        "train_loss": train_loss_list[-1],
                        "test_loss": test_loss_list[-1],
                        "test_accuracy": acc,
                        "test_precision": pre,
                        "test_recall": rec,
                        "test_f1": f1,
                        "lr": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                    })

        if (epoch+1) % 100 == 0 or epoch+1 == args['epochs']:
            torch.save(model, f'models/transformer-{dataset_name}-{epoch+1}.pkl')

