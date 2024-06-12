import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from fontmodel import FontModel
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Executing runner_runner.py on {device}...\n-----------------------------")

    load_model = True
    pretrain_embeddings = False
    pretrain_epochs = 20
    pretrain_lr = 1e-4

    print(f"pretraining hyperparameters:\n\t{pretrain_embeddings=}\n\t{pretrain_epochs=}\n\t{pretrain_lr=}")

    epochs = 0
    batch_size = 256
    lr = 5e-6
    weight_decay=1e-4
    gradient_clip = True

    print(f"training hyperparameters:\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}\n\t{gradient_clip=}")

    vocab_size = 64
    num_layers = 6
    embedding_dim = 512
    num_heads = 8
    ff_dim = 1024

    print(f"fontmodel hyperparameters:\n\t{vocab_size=}\n\t{num_layers=}\n\t{embedding_dim=}\n\t{num_heads=}\n\t{ff_dim=}")

    if load_model:
        model = torch.load('model.pkl')
        model.device = device
    else:
        model = FontModel(
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
    
    dataset_size = 100000
    train_dataset_size = (dataset_size * 4) // 5
    elements_per_seq = 5

    sample_input = torch.randint(0, vocab_size, (dataset_size, elements_per_seq)).to(device)
    sample_truths = torch.remainder(sample_input.sum(dim=-1, keepdim=True) - torch.linspace(0, sample_input.shape[1] - 1, sample_input.shape[1]).to(device), vocab_size).long()
    print(f"{sample_input.shape=}")
    print(f"{sample_truths.shape=}")
    
    out = model(sample_input[:1])
    print(f"{out=}")
    print(f"{out.shape=}")

    print("\nCreating dataset....\n")

    train_tensor_dataset = TensorDataset(sample_input[:train_dataset_size], sample_truths[:train_dataset_size])
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_tensor_dataset = TensorDataset(sample_input[train_dataset_size:], sample_truths[train_dataset_size:])
    test_dataloader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=True)

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
        total_iters=50
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_dataloader):
            inputs = X.to(device)
            truths = y.to(device)

            loss = 0
            for i in range(truths.shape[1]): # Iterate sequence to predict next token
                optimizer.zero_grad()
                out = model(inputs, truths[:,:i]) # Use only output tokens before this truth term
                loss += loss_fn(out, torch.nn.functional.one_hot(truths[:,i:i+1], vocab_size).float()[:,0,:])
                total_loss += loss.item()
            loss.backward()
            # print(torch.sqrt(sum([ torch.norm(p.grad)**2 for p in model.parameters() if p.grad is not None ])))
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        train_loss_list += [total_loss / train_dataset_size]
        
        model.eval()
        total_loss = 0
        for X, y in tqdm(test_dataloader):
            inputs = X.to(device)
            truths = y.to(device)

            for i in range(truths.shape[1]): # Iterate sequence to predict next token
                out = model(inputs, truths[:,:i]) # Use only output tokens before this truth term
                loss = loss_fn(out, torch.nn.functional.one_hot(truths[:,i:i+1], vocab_size).float()[:,0,:])
                total_loss += loss.item()

            # out = model(inputs)
            # loss = loss_fn(out, torch.nn.functional.one_hot(truths, vocab_size).float())
            # total_loss += loss.item()
        test_loss_list += [total_loss / (dataset_size - train_dataset_size)]
        print(f"Epoch {epoch+1}/{epochs} completed. Train Loss = {train_loss_list[-1]};  Test Loss: {test_loss_list[-1]}")
        torch.save(model, 'model.pkl')
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.show()

    print("\nTesting model...\n")

    model.eval()
    num_test = 1000
    test_input = torch.randint(0, vocab_size, (num_test, elements_per_seq)).to(device)
    test_truths = torch.remainder(test_input.sum(dim=-1, keepdim=True) - torch.linspace(0, test_input.shape[1] - 1, test_input.shape[1]).to(device), vocab_size).long()

    test_out = model(test_input, test_truths[:,:0])
    nums = test_out.argmax(dim=-1)

    print(test_truths)
    print(nums)
    print("Accuracy: ")
    print((nums.flatten() == test_truths[:,1].flatten()).sum().item() / num_test)

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