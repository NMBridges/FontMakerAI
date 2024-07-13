import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from diffusion_model import VDM_Encoder, VDM_Decoder
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

    train = True
    test = True
    use_wandb = True
    epochs = 2500
    batch_size = 64
    lr = 4e-4
    weight_decay=1e-2
    gradient_clip = False
    gradient_clip_val = 10.0

    print(f"training hyperparameters:\n\t{use_wandb=}\n\t{epochs=}\n\t{batch_size=}\n\t{lr=}\n\t{weight_decay=}\n\t{gradient_clip=}\n\t{gradient_clip_val=}")

    conv_map = {
        'kernel': (3,3),
        'stride': (1,1),
        'padding': (1,1),
        'dilation': (1,1),
        'down_up_kernel_and_stride': (2,2)
    }

    define_models = True

    T = 1024
    if define_models:
        e = VDM_Encoder(T, device=device).to(device)
        d = VDM_Decoder(T, num_channels=1, label_dim=1, num_classes=52, conv_map=conv_map, device=device).to(device)
    else:
        e = torch.load(f'models/diffusion/model-enc.pkl', map_location=device)
        d = torch.load(f'models/diffusion/model-dec.pkl', map_location=device)
    times = torch.IntTensor(np.linspace(0, T, T+1, dtype=int)).to(device) # TODO: make arange

    base_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(d.parameters(), lr=lr)
    args = {
        "lr": lr,
        "optimizer": optimizer,
        "batch_size": batch_size,
        "epochs": epochs,
        "training": train
    }

    dataset_name = "47000_images_filtered_-1500_1500_(128, 128).pt"
    im_dataset = torch.load(f'./{dataset_name}') / 127.5 - 1.0
    train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
    train_dataloader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=False)

    @torch.no_grad()
    def sampling_traj(d, x_T, T, times, y, num_samples):
        i = T
        x_Ts = [x_T]
        d.eval()
        while i >= 1:
            x_T = d.sample(x_T, times[i:i+1], y)
            i -= 1
            if i % (T // (num_samples - 1)) == 0:
                x_Ts.append(x_T)
        d.train()
        return x_Ts, y

    def show_img_from_tensor(x_T, scale=True):
        new_img = (x_T.cpu().detach().numpy())
        if scale:
            new_img -= new_img.min()
            new_img /= (new_img.max() - new_img.min())
        if new_img.shape[0] == 1:
            plt.imshow(new_img[0], cmap='gray')
            # plt.show()
        else:
            plt.imshow(np.moveaxis(new_img, 0, 2))
            # plt.show()

    if use_wandb:
        wandb.init(
            project="project-typeface",
            config={
                "model_type": "Diffusion",
                "load_model": load_model,
                "pretrain_embeddings": pretrain_embeddings,
                "pretrain_epochs": pretrain_epochs,
                "pretrain_lr": pretrain_lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "gradient_clip": gradient_clip,
                "gradient_clip_val": gradient_clip_val,
                "loss_fn": base_loss_fn.__class__,
                "optimizer": optimizer.__class__,
                "dataset": dataset_name
            }
        )

    losses = []
    d.train()
    for epoch in range(epochs):
        total_loss = 0

        for inp, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            label = label.to(device)
            
            inp = inp.to(device, dtype=torch.float)
            times_i = torch.randint(1, T+1, (inp.shape[0],)).to(device)
            if np.random.random() < 0.1:
                label = None
            
            x_i, eps = e(inp, times_i) # x_{i}, eps_true ~ q(x_{i} | x_{0})
            pred_eps = d(x_i, times_i, label) # eps_theta_{i} ~ p(x_{i-1} | x_{i})

            loss = base_loss_fn(pred_eps, eps)
            total_loss += loss.item() * inp.shape[0]

            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
        total_loss /= len(train_dataloader)
        losses += [total_loss]
        
        if True:
            num_images = 9
            plt.subplot(1, num_images+1, 1)
            plt.plot(losses)
            # plt.show()
            traj, cond = sampling_traj(d, e.reparameterize(torch.zeros(x_i[0:1,:,:,:].shape).to(device), torch.ones(x_i[0:1,:,:,:].shape).to(device))[0], T, times, torch.Tensor([[np.random.randint(1)]]).to(device), num_images)
            for i in range(num_images):
                plt.subplot(1, num_images+1, i+2)
                show_img_from_tensor(traj[i][0])
            plt.show()
            two_five_five = ((traj[num_images-1][0] + 1.0) * 127.5).clamp(0.0, 255.0).round()
            wandb.log({"images": wandb.Image(two_five_five, caption=f"epoch{epoch+1}.png")})

        if use_wandb:
            wandb.log({
                "train_loss": total_loss,
            })

        print(f"Condition: {cond.cpu().detach().numpy()[0]}\n{total_loss=}\nEpoch {epoch+1} finished.")

        torch.save(e, f'models/diffusion/model-enc.pkl')
        torch.save(d, f'models/diffusion/model-dec.pkl')

    torch.save(e, f'models/diffusion/model-enc-MAX-EPOCHS.pkl')
    torch.save(d, f'models/diffusion/model-dec-MAX-EPOCHS.pkl')

