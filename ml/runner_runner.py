import torch
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from fontmodel import FontModel, TransformerDecoder
from dataset_creator import BucketedDataset
from tokenizer import Tokenizer
from glyph_viz import Visualizer
from config import operators
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Executing runner_runner.py on {device}...\n-----------------------------")

    load_model = False
    pretrain_embeddings = False
    pretrain_epochs = 20
    pretrain_lr = 1e-4

    print(f"pretraining hyperparameters:\n\t{pretrain_embeddings=}\n\t{pretrain_epochs=}\n\t{pretrain_lr=}")

    epochs = 2000
    batch_size = 32
    test_batch_size = batch_size // 4
    lr = 5e-6
    weight_decay=1e-4
    gradient_clip = True

    print(f"training hyperparameters:\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}\n\t{gradient_clip=}")

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
    num_layers = 4
    embedding_dim = 256
    num_heads = 8
    ff_dim = 1024

    print(f"fontmodel hyperparameters:\n\t{vocab_size=}\n\t{num_layers=}\n\t{embedding_dim=}\n\t{num_heads=}\n\t{ff_dim=}")

    if load_model:
        model = torch.load('model.pkl')
        model.device = device
    else:
        # model = FontModel(
        #     num_layers=num_layers,
        #     vocab_size=vocab_size,
        #     embedding_dim=embedding_dim,
        #     num_heads=num_heads,
        #     ff_dim=ff_dim,
        #     dropout_rate=0.1,
        #     device=device
        # ).to(device)
        model = TransformerDecoder(
            num_layers=num_layers,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
    # Source: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model solution
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    '''
    BEGIN TEST SECTION
    '''
    
    # dataset_size = 100000
    # train_dataset_size = (dataset_size * 4) // 5
    # elements_per_seq = 5

    # sample_input = torch.randint(0, vocab_size, (dataset_size, elements_per_seq)).to(device)
    # sample_truths = torch.remainder(sample_input.sum(dim=-1, keepdim=True) - torch.linspace(0, sample_input.shape[1] - 1, sample_input.shape[1]).to(device), vocab_size).long()
    # print(f"{sample_input.shape=}")
    # print(f"{sample_truths.shape=}")
    
    # out = model(sample_input[:1])
    # print(f"{out=}")
    # print(f"{out.shape=}")

    # print("\nCreating dataset....\n")

    # train_tensor_dataset = TensorDataset(sample_input[:train_dataset_size], sample_truths[:train_dataset_size])
    # train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    # test_tensor_dataset = TensorDataset(sample_input[train_dataset_size:], sample_truths[train_dataset_size:])
    # test_dataloader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=True)

    train_tensor_dataset = BucketedDataset("./fontmakerai/data_no_subr.csv", tokenizer, (0, 9))
    test_tensor_dataset = BucketedDataset("./fontmakerai/data_no_subr.csv", tokenizer, (9,10))
    dataset_size = len(train_tensor_dataset) + len(test_tensor_dataset)
    train_dataset_size = (dataset_size * 9) // 10
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_tensor_dataset, batch_size=test_batch_size, shuffle=False)

    if pretrain_embeddings:
        print("\nPretraining embeddings...\n")
        tensor_dataset = TensorDataset(torch.arange(vocab_size).reshape((vocab_size, 1)).repeat((512,1)).long(), torch.arange(vocab_size).reshape((vocab_size, 1)).repeat((512, 1)).long())
        pretrain_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        loss_fn = torch.nn.BCELoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr, weight_decay=weight_decay)
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for X, y in tqdm(pretrain_dataloader):
                inputs = X.to(device)
                # truths = y.to(device)
                optimizer.zero_grad()
                out = model.identity_embeddings(inputs)
                loss = loss_fn(out, torch.nn.functional.one_hot(inputs, vocab_size).float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{pretrain_epochs} completed. Total Loss = {total_loss/train_dataset_size}")

    # Extraneous

    train_batch_zeros = torch.zeros((batch_size, 1, embedding_dim)).to(device)
    test_batch_zeros = torch.zeros((test_batch_size, 1, embedding_dim)).to(device)

    # End extraneous

    print("\nTraining model...\n")

    model.train()
    train_loss_list = []
    test_loss_list = []
    loss_fn = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.0001,
        end_factor=1.0,
        total_iters=150
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X in tqdm(train_dataloader):
            loss = 0

            ## FOR ENCODER-DECODER:
            # inputs = X.to(device)
            # truths = y.to(device)
            # for i in range(truths.shape[1]): # Iterate sequence to predict next token
                # optimizer.zero_grad()
                # out = model(inputs, truths[:,:i]) # Use only output tokens before this truth term
                # loss += loss_fn(out, torch.nn.functional.one_hot(truths[:,i:i+1], vocab_size).float()[:,0,:])
                # total_loss += loss.item()
                # loss.backward()
                # if gradient_clip:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                # optimizer.step()
            ## END ENCODER-DECODER

            ## FOR DECODER-ONLY:
            inputs = X.to(device)
            optimizer.zero_grad()
            out = model(train_batch_zeros, inputs[:,:-1]) # Use only output tokens before this truth term
            loss = loss_fn(out, torch.nn.functional.one_hot(inputs.long(), vocab_size).float())
            total_loss += loss.item()
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ## END DECODER-ONLY
        scheduler.step()
        train_loss_list += [total_loss / train_dataset_size]
        
        model.eval()
        total_loss = 0
        for X in tqdm(test_dataloader):
            ## FOR ENCODER-DECODER
            # inputs = X.to(device)
            # truths = y.to(device)
            # for i in range(truths.shape[1]): # Iterate sequence to predict next token
            #     out = model(inputs, truths[:,:i]) # Use only output tokens before this truth term
            #     loss = loss_fn(out, torch.nn.functional.one_hot(truths[:,i:i+1], vocab_size).float()[:,0,:])
            #     total_loss += loss.item()
            ## END ENCODER-DECODER

            ## FOR DECODER-ONLY
            inputs = X.to(device)
            out = model(test_batch_zeros, inputs[:,:-1]) # Use only output tokens before this truth term
            loss = loss_fn(out, torch.nn.functional.one_hot(inputs.long(), vocab_size).float())
            total_loss += loss.item()
            ## END DECODER-ONLY

            # out = model(inputs)
            # loss = loss_fn(out, torch.nn.functional.one_hot(truths, vocab_size).float())
            # total_loss += loss.item()
        test_loss_list += [total_loss / (dataset_size - train_dataset_size)]
        print(f"Epoch {epoch+1}/{epochs} completed. Train Loss = {train_loss_list[-1]};  Test Loss: {test_loss_list[-1]}")
        torch.save(model, 'model.pkl')

        if (epoch + 1) % 10 == 0:
            sequence = model.decode_until_stop(test_batch_zeros[:1], None).cpu().detach().numpy().flatten()
            toks = [tokenizer.reverse_map(tk, use_int=True) for tk in sequence]
            viz = Visualizer(toks[:-1])
            try:
                viz.draw(display=False, filename=f"./training_images/{epoch+1}.png")
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
        ## FOR ENCODER-DECODER
        # inputs = X.to(device)
        # truths = y.to(device)
        # for i in range(truths.shape[1]): # Iterate sequence to predict next token
        #     out = model(inputs, truths[:,:i]) # Use only output tokens before this truth term
        #     loss = loss_fn(out, torch.nn.functional.one_hot(truths[:,i:i+1], vocab_size).float()[:,0,:])
        #     total_loss += loss.item()
        ## END ENCODER-DECODER

        ## FOR DECODER-ONLY
        inputs = X.to(device)
        out = model(torch.zeros((test_batch_size, 1, embedding_dim)).to(device), inputs[:,:-1]).argmax(dim=-1).cpu().detach().numpy() # Use only output tokens before this truth term
        # Zero out all tokens after eos token bc batch decoding doesn't account for this
        out = out * np.pad((out == tokenizer[eos_token]).cumsum(-1) < 1, ((0, 0), (1, 0)), mode='constant', constant_values=True)[:,:-1]
        guesses.append(out)
        truths.append(inputs.cpu().detach().numpy())
        ## END DECODER-ONLY

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
    accuracy = (guesses == truths).sum() / truths.shape[0]
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"\nAccuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1_score}")

    print("Decoding until stop:\n")

    sequence = model.decode_until_stop(torch.zeros((1, 1, embedding_dim)).to(device), None).cpu().detach().numpy().flatten()
    toks = [tokenizer.reverse_map(tk, use_int=True) for tk in sequence]

    print(toks)
    print(f"Length: {len(toks)}")

    with open("glyph_a.txt", 'w') as f:
        j_str = '\', \''
        f.write(f"['{j_str.join(toks[:-1])}']")

    # for test in range(10):
    #     print(f"TEST {test}: (truth = {(elements_per_seq * test) % vocab_size})")
    #     test_val = torch.zeros(1, elements_per_seq, vocab_size)
    #     test_val[0,:,test] = 1.0

    #     print(f"{test_val=}")

    #     test_out = model(test_val)

    #     print(f"{test_out=}")
    #     print(f"{torch.argmax(test_out)}")

    '''
    END TEST SECTION
    '''