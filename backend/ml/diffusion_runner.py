import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from backend.ml.ldm import LDM
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from backend.config import conv_map


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Executing runner_runner.py on {device}...\n-----------------------------")

    args = {
        "load_model": False,
        "train": True,
        "use_wandb": True,
        "vae_epochs": 2500,
        "ddpm_epochs": 2500,
        "vae_batch_size": 128,
        "ddpm_batch_size": 128,
        "vae_lr": 4e-4,
        "ddpm_lr": 4e-4,
        "vae_weight_decay": 1e-5,
        "ddpm_weight_decay": 1e-5,
        "vae_beta": 1e-2,
        "gradient_clip": False,
        "gradient_clip_val": 10.0
    }

    print("Training hyperparameters:")
    pprint(args)

    define_models = True

    T = 1024
    num_glyphs = 26#91

    if args['load_model']:
        model = torch.nn.DataParallel(torch.load(f'models/ldm.pt', map_location=device)).to(device)
    else:
        model = torch.nn.DataParallel(LDM(diffusion_depth=1000, feature_channels=num_glyphs, label_dim=128, conv_map=conv_map)).to(device)

    mse_loss = nn.MSELoss()

    vae_optimizer = torch.optim.AdamW(model.enc_dec.parameters(), lr=args['vae_lr'])
    ddpm_optimizer = torch.optim.AdamW(model.ddpm.parameters(), lr=args['ddpm_lr'])

    dataset_name = "35851allchars_500_500_64_64.pt"
    im_dataset = torch.load(f'./{dataset_name}').to(torch.uint8)[:,:num_glyphs,:,:] / 127.5 - 1.0
    vae_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
    vae_train_dataloader = DataLoader(vae_train_tensor_dataset, batch_size=args['vae_batch_size'], shuffle=False)
    ddpm_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
    ddpm_train_dataloader = DataLoader(ddpm_train_tensor_dataset, batch_size=args['ddpm_batch_size'], shuffle=False)

    @torch.no_grad()
    def sampling_traj(d, q, x_T, T, times, y, num_samples):
        i = T
        x_Ts = [q(x_T)]
        d.eval()
        while i >= 1:
            x_T = d.module.sample(x_T, times[i:i+1], y)
            i -= 1
            if i % (T // (num_samples - 1)) == 0:
                x_Ts.append(q(x_T))
        d.train()
        return x_Ts, y

    def show_img_from_tensor(x_T, scale=True):
        new_img = (x_T.cpu().detach().numpy())
        if scale:
            new_img -= new_img.min()
            new_img /= (new_img.max() - new_img.min())
        if new_img.shape[0] == 1:
            plt.imshow(new_img[0], cmap='gray')
            # plt.savefig("test.png")
            # plt.show()
        else:
            plt.imshow(np.moveaxis(new_img, 0, 2))
            # plt.savefig("test.png")
            # plt.show()

    if args['use_wandb']:
        wandb.init(
            project="project-typeface",
            config={
                "model_type": "Diffusion",
                "load_model": load_model,
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

    train_projectors = True and define_vae
    if train_projectors:
        p_optimizer = torch.optim.AdamW(p.parameters(), lr=lr)
        q_optimizer = torch.optim.AdamW(q.parameters(), lr=lr)
        p.train()
        q.train()
        for epoch in range(pretrain_epochs):
            total_loss = 0
            for inp, label in tqdm(train_dataloader):
                p_optimizer.zero_grad()
                q_optimizer.zero_grad()

                inp = inp.to(device, dtype=torch.float)
                mu, log_sigma2 = p(inp)
                z, eps = p.module.reparameterize(mu, log_sigma2)
                inp_hat = q(z)

                flat_mu = mu.flatten(start_dim=1)
                flat_log_sigma2 = log_sigma2.flatten(start_dim=1)
                kl_div_loss = -0.5 * torch.sum(torch.pow(flat_mu, 2) + torch.exp(flat_log_sigma2) - flat_log_sigma2 - flat_mu.shape[1]) / (flat_mu.shape[0] * flat_mu.shape[1])
                
                loss = base_loss_fn(inp_hat, inp) + 1e-6 * kl_div_loss
                total_loss += loss.item() * inp.shape[0]
                
                loss.backward()
                p_optimizer.step()
                q_optimizer.step()
                torch.cuda.empty_cache()
            total_loss /= len(train_dataloader)
            print(f"{epoch=}: {total_loss=}")
        torch.save(p.module, f'models/vae/model-enc.pkl')
        torch.save(q.module, f'models/vae/model-dec.pkl')
    
    for param in p.parameters():
        param.requires_grad = False
    for param in q.parameters():
        param.requires_grad = False

    losses = []
    p.eval()
    e.eval()
    d.train()
    q.eval()
    for epoch in range(epochs):
        total_loss = 0

        for inp, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            label = label.to(device)
            
            inp = inp.to(device, dtype=torch.float)
            times_i = torch.randint(1, T+1, (inp.shape[0],)).to(device)
            if np.random.random() < 0.1:
                label = None
            
            mu, log_sigma2 = p(inp)
            z, eps = p.module.reparameterize(mu, log_sigma2)
            x_i, eps = e(z, times_i) # x_{i}, eps_true ~ q(x_{i} | x_{0})
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
            shape = z[0:1,:,:,:].shape
            traj, cond = sampling_traj(d, q, e.module.reparameterize(torch.zeros(shape).to(device), torch.ones(shape).to(device))[0], T, times, torch.Tensor([[np.random.randint(1)]]).to(device), num_images)
            for i in range(num_images):
                plt.subplot(1, num_images+1, i+2)
                show_img_from_tensor(traj[i][0,0:1])
            plt.show()
            images = {}
            for i in range(num_glyphs):
                two_five_five = ((traj[num_images-1][0,i] + 1.0) * 127.5).clamp(0.0, 255.0).round()
                images[f"images[{i}]"] = wandb.Image(two_five_five, caption=f"epoch{epoch+1}.png")
            wandb.log(images)

        if use_wandb:
            wandb.log({
                "train_loss": total_loss,
            })

        print(f"Condition: {cond.cpu().detach().numpy()[0]}\n{total_loss=}\nEpoch {epoch+1} finished.")

        torch.save(e.module, f'models/diffusion/model-enc.pkl')
        torch.save(d.module, f'models/diffusion/model-dec.pkl')

    torch.save(e.module, f'models/diffusion/model-enc-MAX-EPOCHS.pkl')
    torch.save(d.module, f'models/diffusion/model-dec-MAX-EPOCHS.pkl')

