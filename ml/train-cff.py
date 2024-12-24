import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from config import conv_map, device, operators

from fontmodel import (FontModel, TransformerDecoder, DecodeInstruction,
                        DecodeType, SamplingType, TransformerScheduler)
from dataset_loader import BucketedDataset
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from performance import PerformanceMetrics
from tablelist_utils import numbers_first, make_non_cumulative

print(f"Executing train-cff.ipynb on {device}...\n-----------------------------")

args = {
    "load_model": False,
    "pretrain_embeddings": True,
    "train_transformer": True,
    "min_number": -1500,
    "max_number": 1500,
    "max_seq_len": 2000,
    "num_layers": 3,
    "embedding_dim": 768,
    "num_heads": 12,
    "ff_dim": 3072,
    "use_wandb": True,
    "pretrain_epochs": 500,
    "pretrain_lr": 4e-4,
    "pretrain_weight_decay": 1e-1,
    "epochs": 5000,
    "batch_size": 64,
    "lr": 1e-4,
    "dropout_rate": 0.1,
    "weight_decay": 1e-5,
    "gradient_clip": False,
    "gradient_clip_val": 5.0,
    "label_smoothing": 0.1,
    "sample_every": 100,
    "use_scheduler": False,
    "scheduler_warmup_steps": 500,
    "data_type": torch.bfloat16
}

print("Training hyperparameters:")
pprint(args)

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
cumulative = False
vocab_size = tokenizer.num_tokens

decode_instr = DecodeInstruction( # NOTE: doesn't matter unless loading from .config.txt fails
    DecodeType.ANCESTRAL,
    SamplingType.GREEDY,
    max_seq_len=args['max_seq_len'],
    k=5,
    p=0,
    temp=0,
    beam_size=6,
)

if args['load_model']:
    model = torch.load(f'models/ldm-basic-35851allchars-0.pkl', map_location=device).to(device)
else:
    model = FontModel(
        num_enc_layers=2,
        num_dec_layers=args['num_layers'],
        vocab_size=vocab_size,
        embedding_dim=args['embedding_dim'],
        num_heads=args['num_heads'],
        ff_dim=args['ff_dim'],
        dropout_rate=args['dropout_rate'],
        device=device
    ).to(device, dtype=args['data_type'])

pretrain_params = [x for x in model.embedder.parameters() if x.requires_grad] + [x for x in model.decoder.token_space.parameters() if x.requires_grad]
pretrain_optimizer = torch.optim.AdamW(pretrain_params, lr=args['pretrain_lr'], weight_decay=args['pretrain_weight_decay'])
params = [x for x in model.decoder.transformer_decoder_layers.parameters() if x.requires_grad]# + [x for x in model.encoder.transformer_encoder_layers.parameters() if x.requires_grad]
optimizer = torch.optim.AdamW(params, lr=args['lr'], weight_decay=args['weight_decay'])

if args['use_scheduler']:
    scheduler = TransformerScheduler(
        optimizer=optimizer,
        dim_embed=args['embedding_dim'],
        warmup_steps=args['scheduler_warmup_steps']
    )

dataset_name = "basic-30513allchars_filtered_centered_scaled.pt"
train_start, train_end = 0, int(0.95 * 30513) * 91
test_start, test_end = train_end, 30513 * 91
cff_dataset = torch.load(f'./{dataset_name}', mmap=True)[train_start:train_end:91]
cff_dataset_test = torch.load(f'./{dataset_name}', mmap=True)[test_start:test_end:91]
cff_train_tensor_dataset = TensorDataset(cff_dataset, torch.zeros(cff_dataset.shape[0], 1))
cff_train_dataloader = DataLoader(cff_train_tensor_dataset, batch_size=args['batch_size'], shuffle=True)
cff_test_tensor_dataset = TensorDataset(cff_dataset_test, torch.zeros(cff_dataset_test.shape[0], 1))
cff_test_dataloader = DataLoader(cff_test_tensor_dataset, batch_size=args['batch_size'], shuffle=True)

if args['use_wandb']:
    wandb.init(
        project="project-typeface",
        config={
            "model_type": "Autoregressive CFF",
            **args
        }
    )

pretrain_loss_fn = torch.nn.CrossEntropyLoss(
    reduction='sum',
    # ignore_index=tokenizer[pad_token],
    # label_smoothing=label_smoothing
)
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

if args["pretrain_embeddings"]:
    print("\nPretraining embeddings...\n")
    tensor_dataset = TensorDataset(torch.arange(vocab_size).reshape((vocab_size, 1)))
    pretrain_dataloader = DataLoader(tensor_dataset, batch_size=vocab_size, shuffle=True)

    model.train()
    for epoch in range(args['pretrain_epochs']):
        total_loss = 0
        for (X,) in tqdm(pretrain_dataloader):
            pretrain_optimizer.zero_grad()
            inputs = X.to(device)
            out = model.identity_embeddings(inputs)
            # loss = pretrain_loss_fn(out.permute(0, 2, 1), inputs)
            loss = numeric_mse_loss(out, inputs)
            total_loss += loss.item()
            loss.backward()
            pretrain_optimizer.step()
            torch.cuda.empty_cache()
        print(f"Epoch {epoch+1}/{args['pretrain_epochs']} completed. Total Loss = {total_loss/vocab_size}")

    model.embedder.weight.requires_grad = False

print("\nTraining model...\n")

if args['train_transformer']:
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    src = torch.zeros((args['batch_size'], 0)).to(device)
    for epoch in range(args['epochs']):
        model.train()
        total_loss = 0
        for (X, _) in tqdm(cff_train_dataloader):
            inputs = X.to(device)
            optimizer.zero_grad()
            # sequence = inputs[0].cpu().detach().numpy().flatten()#.to(device)
            # toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]
            # print(toks[:100])
            out = model(src[:inputs.shape[0]], inputs[:,:-1]) # Use only output tokens before this truth term
            # loss = loss_fn(out.permute(0, 2, 1), inputs.long())
            loss = numeric_mse_loss(out, inputs)
            total_loss += loss.item()
            loss.backward()
            if args['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['gradient_clip_val'])
            optimizer.step()
            if args['use_scheduler']:
                scheduler.step()
            torch.cuda.empty_cache()
        train_loss_list += [total_loss / cff_dataset.shape[0]]
        
        model.eval()
        total_loss = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        with torch.no_grad():
            for (X, _) in tqdm(cff_test_dataloader):
                inputs = X.to(device)
                out = model(src[:inputs.shape[0]], inputs[:,:-1]).permute(0, 2, 1) # Use only output tokens before this truth term
                loss = test_loss_fn(out, inputs.long())
                total_loss += loss.item()

                guesses = out.argmax(dim=1)
                truths = inputs
                true_positives += ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
                false_positives += ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
                true_negatives += ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
                false_negatives += ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
            
            test_loss_list += [total_loss / cff_dataset_test.shape[0]]
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
                    # x = model.encoder(cff_test_tensor_dataset[0:1][0].to(device))[:,0:1,:]
                    x = torch.zeros((1, 0, args['embedding_dim'])).to(device)
                    sequence = model.decode(x, None, decode_instr)[0].cpu().detach().numpy().flatten()
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
                        "lr": args['lr'],
                        "images": img_arr
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
                        "lr": args['lr']
                    })

        if (epoch+1) % 100 == 0:
            torch.save(model, f'models/transformer-{dataset_name}-{epoch+1}.pkl')
