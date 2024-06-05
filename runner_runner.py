import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from fontmodel import FontModel
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Executing runner_runner.py...\n-----------------------------")

    pretrain_embeddings = True
    pretrain_epochs = 20

    print(f"pretraining hyperparameters:\n\t{pretrain_embeddings=}\n\t{pretrain_epochs=}")

    epochs = 150
    batch_size = 128
    lr = 4e-3
    weight_decay=1e-4

    print(f"training hyperparameters:\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}")

    vocab_size = 12
    num_layers = 2
    embedding_dim = 8
    num_heads = 4
    ff_dim = 6

    print(f"fontmodel hyperparameters:\n\t{vocab_size=}\n\t{num_layers=}\n\t{embedding_dim=}\n\t{num_heads=}\n\t{ff_dim=}")

    model = FontModel(
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    '''
    BEGIN TEST SECTION
    '''
    
    dataset_size = 60000
    train_dataset_size = (dataset_size * 4) // 5
    elements_per_seq = 5

    sample_input = torch.randint(0, vocab_size, (dataset_size, elements_per_seq))
    sample_truths = sample_input.sum(dim=-1) % vocab_size
    print(f"{sample_input.shape=}")
    print(f"{sample_truths.shape=}")
    
    out = model(sample_input)
    print(f"{out=}")
    print(f"{out.shape=}")

    print("\nCreating dataset....\n")

    train_tensor_dataset = TensorDataset(sample_input[:train_dataset_size], sample_truths[:train_dataset_size])
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    test_tensor_dataset = TensorDataset(sample_input[train_dataset_size:], sample_truths[train_dataset_size:])
    test_dataloader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=True)

    print("\nPretraining embeddings...\n")
    if pretrain_embeddings:
        tensor_dataset = TensorDataset(torch.arange(vocab_size).reshape((vocab_size, 1)).repeat((512,1)).long(), torch.arange(vocab_size).reshape((vocab_size, 1)).repeat((512, 1)).long())
        pretrain_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        loss_fn = torch.nn.BCELoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for X, y in tqdm(pretrain_dataloader):
                optimizer.zero_grad()
                out = model.identity_embeddings(X)
                loss = loss_fn(out, torch.nn.functional.one_hot(X, vocab_size).float())
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{pretrain_epochs} completed. Total Loss = {total_loss/train_dataset_size}")

    print("\nTraining model...\n")

    model.train()
    train_loss_list = []
    test_loss_list = []
    loss_fn = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, torch.nn.functional.one_hot(y, vocab_size).float())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss_list += [total_loss]
        
        model.eval()
        total_loss = 0
        for X, y in tqdm(test_dataloader):
            out = model(X)
            loss = loss_fn(out, torch.nn.functional.one_hot(y, vocab_size).float())
            total_loss += loss.item()
        test_loss_list += [total_loss]
        print(f"Epoch {epoch+1}/{epochs} completed. Train Loss = {train_loss_list[-1]/train_dataset_size};  Test Loss: {test_loss_list[-1]/(dataset_size - train_dataset_size)}")
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.show()

    print("\nTesting model...\n")

    model.eval()
    num_test = 1000
    test_input = torch.randint(0, vocab_size, (num_test, elements_per_seq))
    test_truths = test_input.sum(dim=-1) % vocab_size

    test_out = model(test_input)
    nums = test_out.argmax(dim=-1)

    print(test_truths)
    print(nums)
    print("Accuracy: ")
    print((nums == test_truths).sum() / num_test)

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