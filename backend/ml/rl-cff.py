#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import wandb
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy

import sys
sys.path.append('/home/ec2-user/FontMakerAI/backend')

from config import device, operators, DecodeType, DecodeInstruction, SamplingType
from ml.tokenizer import Tokenizer
from ml.fontmodel import DecodeInstruction, FontModel
from ml.performance import PerformanceMetrics
from parsing.glyph_viz import Visualizer
from parsing.tablelist_utils import numbers_first, make_non_cumulative


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
    "epochs": 1,
    "batch_size": 4,
    "batch_accumulate": 32,
    "lr": 1e-4,
    "dropout_rate": 0.2,
    "weight_decay": 1e-1,
    "gradient_clip": True,
    "gradient_clip_val": 1.0,
    "label_smoothing": 0.001,
    "sample_every": 1,
    "use_scheduler": False,
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
    "method": "GRPO",
    "grpo_groups": 8,
    "grpo_kl_coeff": 0.04,
}

print("Posttraining hyperparameters:")
pprint(args)


# In[3]:


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


# In[4]:


decode_instr = DecodeInstruction( # NOTE: doesn't matter unless loading from .config.txt fails
    DecodeType.ANCESTRAL,
    SamplingType.TEMPERATURE,
    max_seq_len=args['max_seq_len'],
    k=3,
    p=0,
    temp=1.0,
    beam_size=6,
)


# In[5]:


models_folder = f'../../../models'
if args['load_model']:
    model_pre = torch.load(f'{models_folder}/transformer-basic-33928allchars_centered_scaled_sorted_filtered_cumulative_padded-14.pkl', map_location=device, weights_only=False).to(device)
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
# model = torch.compile(model_pre)
model = deepcopy(model_pre)
original_model = deepcopy(model)
original_model.eval()

del model_pre


# In[6]:


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

dataset_dir = "../../.."
dataset_name = f"{dataset_dir}/basic-33928allchars_centered_scaled_sorted_filtered{'_cumulative' if cumulative else ''}_padded"
train_start, train_end = 0, int(0.95 * max_len) * num_glyphs
test_start, test_end = train_end, max_len * num_glyphs
# max_len = 5
# train_start, train_end = 0, 26*max_len
# test_start, test_end = 0, 26*max_len
cff_dataset = torch.load(f'./{dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
cff_dataset_test = torch.load(f'./{dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
im_dataset_name = f"{dataset_dir}/basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)"
im_dataset = torch.load(f'./{im_dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
im_dataset_test = torch.load(f'./{im_dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
cff_train_tensor_dataset = TensorDataset(cff_dataset, im_dataset)
cff_train_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['batch_size'], shuffle=True)
cff_test_tensor_dataset = TensorDataset(cff_dataset_test, im_dataset_test)
cff_test_dataloader = DataLoader(cff_test_tensor_dataset, batch_size=args['batch_size'] * 4, shuffle=True)


# In[ ]:


print("\nPost-training model...\n")

kl_loss = torch.nn.functional.kl_div

@torch.no_grad()
def value_fns(image_gt, output_tokens, method="PPO"):
    '''
    image_gt: the ground truth image
    output_tokens: the model's predicted output
    '''
    try:
        sequence = output_tokens.cpu().detach().numpy().flatten()
        toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence]
        toks = [tok for tok in toks if tok != '<PAD2>' and tok != '<PAD>']
        if cumulative:
            toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
        else:
            toks = numbers_first(toks, tokenizer, return_string=False)
        viz = Visualizer(toks)
        im_pixel_size = (128, 128)
        crop_factor = 1
        dpi = 1
        boundaries = (int((im_pixel_size[0] * (crop_factor * 100 / dpi - 1)) // 2), int((im_pixel_size[1] * (crop_factor * 100 / dpi - 1)) // 2))
        im_size_inches = ((im_pixel_size[0] * crop_factor) / dpi, (im_pixel_size[1] * crop_factor) / dpi)
        if method == "PPO":
            output_images = viz.rl_draw(
                im_size_inches=im_size_inches,
                bounds=(-300, 300),
                dpi=dpi
            ) / 255.0
        elif method == "GRPO":
            output_images = viz.draw(
                display=False,
                filename=None,
                return_image=True,
                center=False,
                im_size_inches=im_size_inches,
                bounds=(-300, 300),
                dpi=dpi
            )[None,:,:,0] / 255.0
    except Exception as e:
        print(f"Error drawing image: {e} for tokens {toks}")
        if method == "PPO":
            output_images = np.ones((sequence.shape[-1] // 7, 128, 128))
        elif method == "GRPO":
            output_images = np.ones((1, 128, 128))
    value = -torch.abs((image_gt + 1) / 2 - torch.Tensor(output_images).to(device))
    value = value.mean(dim=(1,2))
    return value


# In[8]:


def decode(epoch : int, batch_idx : int):
    model.eval()
    with torch.no_grad():
        try:
            flag = True
            idx = np.random.randint(0, im_dataset_test.shape[0])
            im = im_dataset_test[idx:idx+1].to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            new_decode_instr = deepcopy(decode_instr)
            new_decode_instr.sampling_type = SamplingType.GREEDY
            sequence = model.decode(im, None, new_decode_instr)[0].cpu().detach().numpy().flatten()
            if len(sequence) == decode_instr.max_seq_len:
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence] + ['endchar']
            else:
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

            print("Before:", toks)
            toks = [tok for tok in toks if tok != '<PAD2>' and tok != '<PAD>']
            if cumulative:
                toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
                # toks = make_non_cumulative(toks, tokenizer)
            else:
                toks = numbers_first(toks, tokenizer, return_string=False)
                # toks = toks
            print("After:", toks)
            viz = Visualizer(toks)

            if toks[2] != "rmoveto" and toks[3] != "rmoveto":
                raise Exception("first operator is not rmoveto")
            
            im_pixel_size = (128, 128)
            crop_factor = 1
            dpi = 1
            boundaries = (int((im_pixel_size[0] * (crop_factor * 100 / dpi - 1)) // 2), int((im_pixel_size[1] * (crop_factor * 100 / dpi - 1)) // 2))
            im_size_inches = ((im_pixel_size[0] * crop_factor) / dpi, (im_pixel_size[1] * crop_factor) / dpi)
            img_arr = viz.draw(
                display=False,
                filename=None,
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
        print("Decoding failed.")
        return (None, None)
        # return (wandb.Image(im[0].to(device=device, dtype=torch.float32).cpu().detach().numpy(), caption=f"epoch{epoch+1}_{batch_idx+1}.png"), None)


# In[9]:


def test_model():
    training = model.training
    model.eval()
    total_loss = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[pad_token],
        label_smoothing=0.0
    )
    with torch.no_grad():
        for idx, (X, im) in enumerate(tqdm(cff_test_dataloader)):
            inputs = X.to(device, dtype=torch.int32)
            im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0
            out = model(im, inputs[:,:-7])

            loss = loss_fn(out.permute(0, 2, 1), inputs.long()) / X.shape[0]
            # loss = numeric_mse_loss(out, inputs) / X.shape[0]
            
            total_loss += loss.item() * X.shape[0]
            torch.cuda.empty_cache()

            # guesses = out.permute(0, 2, 1).argmax(dim=1)
            # truths = inputs
            # true_positives += ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
            # false_positives += ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
            # true_negatives += ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
            # false_negatives += ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
        
        test_loss = total_loss / ((idx+1)*args['batch_size']*4)
        # acc, pre, rec, f1 = PerformanceMetrics.all_metrics(
        #     tp=true_positives,
        #     fp=false_positives,
        #     tn=true_negatives,
        #     fn=false_negatives
        # )
    model.train(training)
    return test_loss


# In[ ]:


if args['use_wandb']:
    wandb.init(
        project="project-typeface",
        config={
            "model_type": "Autoregressive CFF",
            **args
        }
    )


# In[ ]:


wandb.log({
    'test_loss': test_model()
})

for epoch in range(args['epochs']):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    last_loss = 0
    train_batches = (max_len*(num_glyphs // step_every) // args['batch_size']) + 1
    for idx, (X, im) in enumerate((cff_train_dataloader)):
        if idx >= train_batches:
            break
        inputs = X.to(device, dtype=torch.int32)
        im = im.to(dtype=args['data_type'], device=device).unsqueeze(1) / 127.5 - 1.0

        if args['method'] == "GRPO":
            inputs = inputs.repeat_interleave(args["grpo_groups"], 0)
            im = im.repeat_interleave(args["grpo_groups"], 0)

        # output tokens from the current model
        with torch.no_grad():
            print(f"[step {(idx // args['batch_accumulate'])+1}-{(idx % args['batch_accumulate'])+1}/{args['batch_accumulate']}] Decoding {inputs.shape[0]} images...", end='')
            model.eval()
            out_tokens = model.rl_decode(im, None, decode_instr)
            model.train()
            print(" done.")
        if out_tokens.shape[1] != decode_instr.max_seq_len:
            out_tokens = out_tokens[:,:-1]
        in_tokens = out_tokens[:,:-7] # note: SOS token is prepended in forward()

        # ADVANTAGES
        if args['method'] == "PPO":
            gamma_ = 0.99
            lambda_ = 0.95
            values = torch.zeros((out_tokens.shape[0], out_tokens.shape[1]//7-1)).to(device, dtype=args['data_type'])
            for b in tqdm(range(out_tokens.shape[0])):
                index_of_endchar = (out_tokens[b,:] != model.decoder.pad_token[0,0]).nonzero(as_tuple=True)[0][-1].item() + 7
                if index_of_endchar == decode_instr.max_seq_len - 1:
                    index_of_endchar += 1
                values[b,:index_of_endchar//7-2] = value_fns(im[b], out_tokens[b,:index_of_endchar], method=method)[2:]

            deltas = torch.Tensor([[gamma_ * values[b][i+1] - values[b][i] for i in range(values.shape[1]-1)] for b in range(out_tokens.shape[0])]).to(device, dtype=args['data_type']) # (batch_size, seq_len=5040)
            adv = torch.zeros_like(deltas)
            for i in range(adv.shape[1]):
                if i == 0:
                    adv[:,adv.shape[1]-i-1] = deltas[:,adv.shape[1]-i-1]
                else:
                    adv[:,adv.shape[1]-i-1] = deltas[:,adv.shape[1]-i-1] + gamma_ * lambda_ * adv[:,adv.shape[1]-i]
            adv = adv.repeat_interleave(7, dim=1) # (batch_size, seq_len=5040)
        elif args['method'] == "GRPO":
            rewards = torch.zeros((out_tokens.shape[0],)).to(device, dtype=args['data_type'])
            for b in tqdm(range(out_tokens.shape[0])):
                index_of_endchar = (out_tokens[b,:] != model.decoder.pad_token[0,0]).nonzero(as_tuple=True)[0][-1].item() + 7
                if index_of_endchar == decode_instr.max_seq_len - 1:
                    index_of_endchar += 1
                rewards[b] = value_fns(im[b], out_tokens[b,:index_of_endchar], method=args['method'])
            adv = rewards.unflatten(0, (-1, args["grpo_groups"]))
            adv = (adv - adv.mean(dim=1, keepdim=True)) / adv.std(dim=1, keepdim=True)
            adv = adv.flatten().unsqueeze(-1).repeat((1, out_tokens.shape[1]))
        
        if args['method'] == "PPO":
            k = 1
        elif args['method'] == "GRPO":
            print(f"[step {(idx // args['batch_accumulate'])+1}-{(idx % args['batch_accumulate'])+1}/{args['batch_accumulate']}] Getting reference distribution...", end='')
            with torch.no_grad():
                log_dist_original = original_model(im, in_tokens).log_softmax(dim=-1)
            print(" done.")
            log_prob_original = torch.gather(log_dist_original, dim=-1, index=out_tokens.unsqueeze(1)).squeeze(1)

            print(f"[step {(idx // args['batch_accumulate'])+1}-{(idx % args['batch_accumulate'])+1}/{args['batch_accumulate']}] Getting previous distribution...", end='')
            with torch.no_grad():
                log_dist_prev = model(im, in_tokens).log_softmax(dim=-1)
            print(" done.")
            log_prob_prev = torch.gather(log_dist_prev, dim=-1, index=out_tokens.unsqueeze(1)).squeeze(1)
            k = 1

        for _ in range(k):
            # token distributions
            print(f"[step {(idx // args['batch_accumulate'])+1}-{(idx % args['batch_accumulate'])+1}/{args['batch_accumulate']}] Getting new distribution...", end='')
            log_dist_new = model(im, in_tokens).log_softmax(dim=-1)
            print(" done.")
            
            log_prob_new = torch.gather(log_dist_new, dim=-1, index=out_tokens.unsqueeze(1)).squeeze(1)
            rel_prob = (log_prob_new - log_prob_prev).exp() # (batch_size, seq_len)
            eps = 0.2
            if args['method'] == "PPO":
                loss = -(torch.minimum(rel_prob[:,14:] * adv, torch.clip(rel_prob[:,14:], 1-eps, 1+eps) * adv) * (out_tokens[:,14:] != tokenizer[pad_token])).mean()
            elif args['method'] == "GRPO":
                kl_approx = ((log_prob_original - log_prob_new).exp() - (log_prob_original - log_prob_new) - 1) * (out_tokens != tokenizer[pad_token])
                loss = -(torch.minimum(rel_prob * adv, torch.clip(rel_prob, 1-eps, 1+eps) * adv) * (out_tokens != tokenizer[pad_token])).mean() + args["grpo_kl_coeff"] * kl_approx.mean()

            print(f"[step {(idx // args['batch_accumulate'])+1}-{(idx % args['batch_accumulate'])+1}/{args['batch_accumulate']}] Propagating loss...", end='')

            total_loss += loss.item() * X.shape[0]
            loss.backward()

            if k > 1 or (idx+1) % args['batch_accumulate'] == 0:
                if args['gradient_clip']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args['gradient_clip_val'])
                print(" stepping...", end='')
                optimizer.step()
                optimizer.zero_grad()
                if args['use_scheduler']:
                    scheduler.step()
            
            print(" done.")

        diff = total_loss - last_loss
        last_loss = total_loss

        if args['use_wandb']:
            if (idx+1) % (25 * args['batch_accumulate']) == 0:
                wandb.log({
                    'test_loss': test_model()
                })
            if (idx+1) % (2 * args['batch_accumulate']) == 0 or (idx == train_batches-1 and (epoch+1) % args['sample_every'] == 0):
                goal_image, img_arr = decode(epoch, idx)
                wandb.log({
                    "posttrain_goal_image": goal_image,
                    "posttrain_images": img_arr,
                    "posttrain_loss_step": diff / (args['batch_accumulate'] * args['batch_size']),
                    "posttrain_lr_step": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                })
            elif (idx+1) % args['batch_accumulate'] == 0:
                wandb.log({
                    "posttrain_loss_step": diff / (args['batch_accumulate'] * args['batch_size']),
                    "posttrain_lr_step": args['lr'] if not args['use_scheduler'] else scheduler.get_last_lr()[0],
                })

    # if (epoch+1) % 100 == 0 or epoch+1 == args['epochs']:
    # if max_len > 100:
    #     torch.save(model, f'models/transformer-{dataset_name}-{epoch+1}.pkl')

