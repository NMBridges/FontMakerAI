import torch
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from fontmodel import (FontModel, TransformerDecoder, DecodeInstruction,
                        DecodeType, SamplingType, TransformerScheduler)
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from performance import PerformanceMetrics
from tablelist_utils import numbers_first, make_non_cumulative
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
    pretrain_embeddings = False
    pretrain_epochs = 100
    pretrain_lr = 1e-4

    print(f"pretraining hyperparameters:\n\t{pretrain_embeddings=}\n\t{pretrain_epochs=}\n\t{pretrain_lr=}")

    use_wandb = True
    epochs = 1000
    batch_size = 32
    test_batch_size = batch_size // 4
    lr = 3e-7
    weight_decay=1e-4
    gradient_clip = False
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
    ff_dim = 1024
    decode_instr = DecodeInstruction(
        DecodeType.ANCESTRAL,
        SamplingType.GREEDY,
        max_seq_len=2000,
        k=5,
        p=0,
        temp=1,
        beam_size=6,
    )

    print(f"fontmodel hyperparameters:\n\t{vocab_size=}\n\t{num_layers=}\n\t{embedding_dim=}\n\t{num_heads=}\n\t{ff_dim=}")
    
    if load_model:
        model = torch.load('./fontmakerai/model.pkl', map_location=device).to(device)
        model.device = device
    else:
        model = TransformerDecoder(
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)

    dataset_name = '46918_fonts.csv'

    # Loss constants
    curve_width = 7
    first_numeric_idx = 3 + len(tokenizer.possible_operators)
    hlf = curve_width // 2
    offset = torch.arange(-hlf, hlf+1, dtype=torch.long)[None,None,:].to(device) # (1, 1, curve_width)
    neg_offset_exp = (-offset.abs().unsqueeze(-1)).exp()
    kl_loss = torch.nn.KLDivLoss(reduction='sum')
    cross_entropy_loss = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[tokenizer.pad_token],
        label_smoothing=label_smoothing
    )

    def numeric_mse_loss(output : torch.Tensor, targets : torch.Tensor):
        # targets : (batch_size, seq_len)
        with torch.no_grad():
            is_numeric_tgt = (targets >= first_numeric_idx).unsqueeze(-1) # (batch_size, seq_len, 1)
            is_non_padding = (targets > 0).unsqueeze(-1) # (batch_size, seq_len, 1)

            single_tgt = torch.nn.functional.one_hot(targets, tokenizer.num_tokens) # (batch_size, seq_len, vocab_size)
            multi_tgt = torch.clip(targets.unsqueeze(-1) + offset, min=1, max=tokenizer.num_tokens-1) # (batch_size, seq_len, curve_width)
            multi_tgt = torch.nn.functional.one_hot(multi_tgt, tokenizer.num_tokens) * neg_offset_exp # (batch_size, seq_len, curve_width, vocab_size)
            multi_tgt[:,:,:,:first_numeric_idx] = 0
            multi_tgt = multi_tgt.sum(dim=-2) / (multi_tgt.sum(dim=(-2,-1)).unsqueeze(-1) + 1e-10) # (batch_size, seq_len, vocab_size)
            tgt_dist = is_numeric_tgt * multi_tgt + (~is_numeric_tgt) * (is_non_padding) * single_tgt # (batch_size, seq_len, vocab_size)

        return kl_loss(output.log_softmax(dim=-1), tgt_dist)# + cross_entropy_loss(output, targets)


    pretrain_loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[pad_token],
        label_smoothing=label_smoothing
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[pad_token],
        label_smoothing=label_smoothing
    )
    test_loss_fn = torch.nn.CrossEntropyLoss(
        reduction='sum',
        ignore_index=tokenizer[pad_token],
        label_smoothing=0.0
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer=optimizer,
    #     start_factor=0.0001,
    #     end_factor=1.0,
    #     total_iters=4000
    # )
    scheduler = TransformerScheduler(
        optimizer=optimizer,
        dim_embed=embedding_dim,
        warmup_steps=4000
    )

    print(f"optimization hyperparameters:\n\t{loss_fn=}\n\t{optimizer=}\n\t{scheduler=}")

    if use_wandb:
        wandb.init(
            project="project-typeface",
            config={
                "model_type": "Transformer",
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
                "scheduler": scheduler,
                "dataset": dataset_name
            }
        )

    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model solution
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    '''
    BEGIN TEST SECTION
    '''

    print("Loading dataset...")

    if epochs > 0:
        train_tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (0,-3))
        train_dataset_size = len(train_tensor_dataset)
        train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=False)
    
    test_tensor_dataset = BucketedDataset(f"./fontmakerai/{dataset_name}", tokenizer, (-3,-1))
    test_dataset_size = len(test_tensor_dataset)
    test_dataloader = DataLoader(test_tensor_dataset, batch_size=test_batch_size, shuffle=False)
    
    if pretrain_embeddings:
        print("\nPretraining embeddings...\n")
        tensor_dataset = TensorDataset(torch.arange(vocab_size).reshape((vocab_size, 1)).long())
        pretrain_dataloader = DataLoader(tensor_dataset, batch_size=vocab_size // 8, shuffle=True)

        model.train()
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for (X,) in tqdm(pretrain_dataloader):
                inputs = X.to(device)
                pretrain_optimizer.zero_grad()
                out = model.identity_embeddings(inputs).permute(0, 2, 1)
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

    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X in tqdm(train_dataloader):
            inputs = X.to(device)
            optimizer.zero_grad()
            out = model(train_batch_zeros, inputs[:,:-1])#.permute(0, 2, 1) # Use only output tokens before this truth term
            # loss = loss_fn(out, inputs)
            loss = numeric_mse_loss(out, inputs)
            total_loss += loss.item()
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            scheduler.step()
        train_loss_list += [total_loss / train_dataset_size]
        
        model.eval()
        total_loss = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        with torch.no_grad():
            for X in tqdm(test_dataloader):
                inputs = X.to(device)
                out = model(test_batch_zeros, inputs[:,:-1]).permute(0, 2, 1) # Use only output tokens before this truth term
                loss = test_loss_fn(out, inputs)
                total_loss += loss.item()

                guesses = out.argmax(dim=1)
                truths = inputs
                true_positives += ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
                false_positives += ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
                true_negatives += ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
                false_negatives += ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
            
            test_loss_list += [total_loss / test_dataset_size]
            acc, pre, rec, f1 = PerformanceMetrics.all_metrics(
                tp=true_positives,
                fp=false_positives,
                tn=true_negatives,
                fn=false_negatives
            )
            if use_wandb:
                wandb.log({
                    "train_loss": train_loss_list[-1],
                    "test_loss": test_loss_list[-1],
                    "test_accuracy": acc,
                    "test_precision": pre,
                    "test_recall": rec,
                    "test_f1": f1,
                    "lr": scheduler.get_lr()[0]
                })
            print(f"Epoch {epoch+1}/{epochs} completed. Train Loss = {train_loss_list[-1]};  Test Loss: {test_loss_list[-1]}")
            torch.save(model, './fontmakerai/model.pkl')
        
            if (epoch + 1) % 5 == 0:
                sequence = model.decode(test_batch_zeros[:1], None, decode_instr).cpu().detach().numpy().flatten()
                toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence[:-1]]

                print("Before:", toks)
                try:
                    with open(f"./fontmakerai/training_images/{epoch+1}.txt", 'w') as f:
                        j_str = '\', \''
                        f.write(f"Before: ['{j_str.join([str(x) for x in toks])}']\n\n")
                    toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
                    print("After:", toks)
                    viz = Visualizer(toks)
                    with open(f"./fontmakerai/training_images/{epoch+1}.txt", 'a', newline='\n') as f:
                        j_str = '\', \''
                        f.write(f"After: ['{j_str.join([str(x) for x in toks[:-1]])}']")
                    viz.draw(display=False, filename=f"./fontmakerai/training_images/{epoch+1}.png")
                except Exception as e:
                    print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")
    
    if train_loss_list:
        plt.plot(train_loss_list)
        plt.plot(test_loss_list)
        plt.show()

    print("\nTesting model...\n")

    model.eval()
    guesses = []
    truths = []
    for X in tqdm(test_dataloader):
        inputs = X.to(device)
        out = model(test_batch_zeros, inputs[:,:-1]).argmax(dim=-1).cpu().detach().numpy() # Use only output tokens before this truth term
        # Zero out all tokens after eos token bc batch decoding doesn't account for this
        out = out * np.pad((out == tokenizer[eos_token]).cumsum(-1) < 1, ((0, 0), (1, 0)), mode='constant', constant_values=True)[:,:-1]
        guesses.append(out)
        truths.append(inputs.cpu().detach().numpy())

    guesses = np.array(guesses).flatten()
    truths = np.array(truths).flatten()
    true_positives = ((guesses == truths) * (truths != tokenizer[pad_token])).sum()
    false_positives = ((guesses != truths) * (truths == tokenizer[pad_token])).sum()
    true_negatives = ((guesses == truths) * (truths == tokenizer[pad_token])).sum()
    false_negatives = ((guesses != truths) * (truths != tokenizer[pad_token])).sum()
    print(truths)
    print(guesses)
    print(f"\n{true_positives=}")
    print(f"{false_positives=}")
    print(f"{true_negatives=}")
    print(f"{false_negatives=}")
    accuracy = PerformanceMetrics.accuracy(true_positives, false_positives, true_negatives, false_negatives)
    precision = PerformanceMetrics.precision(true_positives, false_positives, true_negatives, false_negatives)
    recall = PerformanceMetrics.recall(true_positives, false_positives, true_negatives, false_negatives)
    f1_score = PerformanceMetrics.f1(true_positives, false_positives, true_negatives, false_negatives)
    print(f"\nAccuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")

    print("Decoding until stop:\n")

    sequence = model.decode(test_batch_zeros[:1], None, decode_instr).cpu().detach().numpy().flatten()
    toks = [tokenizer.reverse_map(tk.item(), use_int=True) for tk in sequence]
    print(f"Test Before: {toks}")
    try:
        toks = numbers_first(make_non_cumulative(toks, tokenizer), tokenizer, return_string=False)
        print("Test After:", toks)
    except:
        print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")

    viz = Visualizer(toks)

    print(f"Length: {len(toks)}")

    with open("glyph_a.txt", 'w') as f:
        j_str = '\', \''
        f.write(f"['{j_str.join([str(x) for x in toks])}']")

    try:
        viz.draw(display=False, filename='./fontmakerai/training_images/out.png')
    except Exception as e:
        print(f"Could not generate visualization; generated output was not formatted correctly: {e.args[0]}")

    '''
    END TEST SECTION
    '''