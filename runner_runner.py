import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from fontmodel import FontModel
from tqdm import tqdm

if __name__ == "__main__":
    print("Executing runner_runner.py...\n-----------------------------")

    epochs = 250
    batch_size = 16
    lr = 2e-3
    weight_decay=1e-10

    print(f"training hyperparameters:\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}")

    vocab_size = 12
    num_layers = 4
    embedding_dim = 6
    num_heads = 2
    ff_dim = 4

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
    
    dataset_size = 10000
    elements_per_seq = 5

    sample_input = torch.randint(0, vocab_size, (dataset_size, elements_per_seq))
    sample_truths = sample_input.sum(dim=-1) % vocab_size
    print(f"{sample_input.shape=}")
    print(f"{sample_truths.shape=}")
    
    out = model(sample_input)
    print(f"{out=}")
    print(f"{out.shape=}")

    print("\nCreating dataset....\n")

    tensor_dataset = TensorDataset(sample_input, sample_truths)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    print("\nTraining model...\n")

    model.train()
    loss_fn = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        total_loss = 0
        for X, y in tqdm(dataloader):
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, torch.nn.functional.one_hot(y, vocab_size).float())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed. Total Loss = {total_loss/dataset_size}")

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