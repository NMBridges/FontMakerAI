import torch
import torch.nn as nn
from diffusion_model import FontDiffusionModel
from tokenizer import Tokenizer
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from fontmodel import (DecodeInstruction,
                        DecodeType, SamplingType, TransformerScheduler)
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from performance import PerformanceMetrics
from config import operators
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Executing runner_runner.py on {device}...\n-----------------------------")

    load_model = False
    pretrain_embeddings = True
    pretrain_epochs = 100
    pretrain_lr = 1e-3

    print(f"pretraining hyperparameters:\n\t{pretrain_embeddings=}\n\t{pretrain_epochs=}\n\t{pretrain_lr=}")

    use_wandb = True
    epochs = 1000
    batch_size = 64
    test_batch_size = batch_size // 4
    lr = 2e-4
    weight_decay=1e-5
    gradient_clip = True
    gradient_clip_val = 10.0
    label_smoothing = 0.1

    print(f"training hyperparameters:\n\t{use_wandb=}\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}\n\t{gradient_clip=}\n\t{gradient_clip_val=}")

    min_number = -1500
    max_number = 1500
    pad_token = "<PAD>"
    sos_token = "<SOS>"
    eos_token = "<EOS>"
    tokenizer = Tokenizer(
        min_number=min_number,
        max_number=max_number,
        possible_operators=operators,
        pad_token=pad_token,
        sos_token=sos_token,
        eos_token=eos_token
    )

    print(f"tokenizer hyperparameters:\n\t{min_number=}\n\t{max_number=}\n\t{tokenizer.num_tokens=}\n\t{pad_token=}\n\t{sos_token=}\n\t{eos_token=}")

    vocab_size = tokenizer.num_tokens
    num_layers = 6
    embedding_dim = 512
    num_heads = 8
    ff_dim = 2048
    decode_instr = DecodeInstruction(
        DecodeType.ANCESTRAL,
        SamplingType.TEMPERATURE,
        max_seq_len=500,
        k=5,
        p=0,
        temp=2,
        beam_size=6,
    )

    print(f"fontmodel hyperparameters:\n\t{vocab_size=}\n\t{num_layers=}\n\t{embedding_dim=}\n\t{num_heads=}\n\t{ff_dim=}")
        
    T = 1000
    model = FontDiffusionModel(
        depth=T,
        num_layers=4,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        embedder=None,
        dropout_rate=0.1,
        device=device
    )


    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model solution
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    print("Loading dataset...")

    dataset_name = "cleaned_cff_data_june28.csv"
    train_tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (0,-2))
    test_tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (-2,-1))
    dataset_size = len(train_tensor_dataset) + len(test_tensor_dataset)
    train_dataset_size = (dataset_size * 9) // 10
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_tensor_dataset, batch_size=test_batch_size, shuffle=False)
    pretrain_loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[pad_token],
        label_smoothing=label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=weight_decay
    )
    scheduler = TransformerScheduler(
        optimizer=optimizer,
        dim_embed=embedding_dim,
        warmup_steps=4000
    )
    loss_fn = nn.L1Loss()

    if use_wandb:
        wandb.init(
            project="project-typeface",
            config={
                "model_type": "Diffusion (Transformer Backbone)",
                "load_model": load_model,
                "pretrain_embeddings": pretrain_embeddings,
                "pretrain_epochs": pretrain_epochs,
                "pretrain_lr": pretrain_lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "test_batch_size": test_batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "gradient_clip": gradient_clip,
                "gradient_clip_val": gradient_clip_val,
                "label_smoothing": label_smoothing,
                "min_number": min_number,
                "max_number": max_number,
                "pad_token": pad_token,
                "sos_token": sos_token,
                "eos_token": eos_token,
                "possible_operators": operators,
                "vocab_size": vocab_size,
                "num_layers": num_layers,
                "embedding_dim": embedding_dim,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "model_class": model.__class__,
                "loss_fn": loss_fn.__class__,
                "optimizer": optimizer.__class__,
                "scheduler": "NONE",
                "dataset": dataset_name
            }
        )

    if pretrain_embeddings:
        print("\nPretraining embeddings...\n")
        tensor_dataset = TensorDataset(torch.arange(vocab_size).reshape((vocab_size, 1)))
        pretrain_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for (X,) in tqdm(pretrain_dataloader):
                inputs = X.to(device)
                pretrain_optimizer.zero_grad()
                out = model.token_space(model.embedder(inputs)).permute(0, 2, 1)
                loss = pretrain_loss_fn(out, inputs)
                total_loss += loss.item()
                loss.backward()
                pretrain_optimizer.step()
            print(f"Epoch {epoch+1}/{pretrain_epochs} completed. Total Loss = {total_loss/train_dataset_size}")

    # Extraneous

    train_batch_zeros = torch.zeros((batch_size, 1, embedding_dim)).to(device)
    test_batch_zeros = torch.zeros((test_batch_size, 1, embedding_dim)).to(device)

    # End extraneous

    print("\nTraining model...\n")

    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        total_loss = 0

        model.train()
        for X in tqdm(train_dataloader):
            inputs = X.to(device)
            optimizer.zero_grad()
            
            times_i = torch.randint(1, T+1, (inputs.shape[0],)).to(device)
            
            embeddings = model.embedder(inputs)
            x_i, eps = model.encode(embeddings, times_i) # x_{i}, eps_true ~ q(x_{i} | x_{0})
            pred_eps = model.decoder.noise_prediction(x_i, times_i) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
            
            loss = loss_fn(pred_eps, eps)
            total_loss += loss.item() * inputs.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

        test_total_loss = 0
        model.eval()
        with torch.no_grad():
            for X in tqdm(test_dataloader):
                inputs = X.to(device)
                times_i = torch.randint(1, T+1, (inputs.shape[0],)).to(device)
                embeddings = model.embedder(inputs)
                x_i, eps = model.encode(embeddings, times_i) # x_{i}, eps_true ~ q(x_{i} | x_{0})
                pred_eps = model.decode(x_i, times_i) # eps_theta_{i} ~ p(x_{i-1} | x_{i})
                loss = loss_fn(pred_eps, eps)
                test_total_loss += loss.item()
            
            if use_wandb:
                wandb.log({
                    "train_loss": total_loss / train_dataset_size,
                    "test_loss": test_total_loss / (dataset_size - train_dataset_size),
                    "lr": scheduler.get_lr()[0]
                })
            print(f"Epoch {epoch+1}/{epochs} completed. Train Loss = {total_loss / train_dataset_size};  Test Loss: {test_total_loss / (dataset_size - train_dataset_size)}")
            torch.save(model, './fontmakerai/model.pkl')
        
            if (epoch + 1) % 4 == 0:
                # (batch_size, seq_len, embed_dim)
                sequence, traj = model.sample((1, 100, embedding_dim))
                sequence = model.token_space(sequence).argmax(dim=-1) # (batch_size, seq_len)
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence.cpu().detach().numpy().flatten()]

                print("Before:", toks[:-1])
                try:
                    ops = []
                    nums = []
                    for col in range(len(toks)):
                        if toks[col] in tokenizer.possible_operators:
                            ops.append(toks[col])
                            nums.append([])
                        else:
                            if len(nums) == 0:
                                raise Exception("Generated 'table list' cannot start with a non-operator")
                            nums[-1].append(toks[col])
                    toks = []
                    i = 0
                    j = 0
                    while i < len(ops) or j < len(nums):
                        if j < len(nums):
                            toks += nums[j]
                            j += 1
                        if i < len(ops):
                            toks.append(ops[i])
                            i += 1
                    print("After:", toks[:-1])
                    viz = Visualizer(toks[:-1])
                    with open(f"./fontmakerai/training_images/{epoch+1}.txt", 'w') as f:
                        j_str = '\', \''
                        f.write(f"['{j_str.join([str(x) for x in toks[:-1]])}']")
                    viz.draw(display=False, filename=f"./fontmakerai/training_images/{epoch+1}.png")
                except Exception as e:
                    print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")
