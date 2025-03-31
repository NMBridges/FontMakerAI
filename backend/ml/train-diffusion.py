import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, dataset
import numpy as np
from backend.ml.ldm import LDM
from backend.ml.ddpm import DDPM
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from backend.config import conv_map, device

print(f"Executing train-diffusion.ipynb on {device}...\n-----------------------------")

args = {
    "load_model": False,
    "train_vae": True,
    "train_ddpm": True,
    "use_wandb": True,
    "vae_epochs": 100,
    "ddpm_epochs": 2500,
    "vae_batch_size": 26 * 16,
    "ddpm_batch_size": 26 * 64,
    "vae_lr": 4e-4,
    "ddpm_lr": 4e-4,
    "vae_weight_decay": 1e-5,
    "ddpm_weight_decay": 1e-5,
    "vae_beta": 1e-0,
    "gradient_clip": True,
    "gradient_clip_val": 1.0,
    "T": 1024,
    "num_glyphs": 26,
    "label_dim": 128,
    "loss_fn": nn.MSELoss(),
    "precision": torch.float32,
    "rescale_latent": False,
}

print("Training hyperparameters:")
pprint(args)

if args['load_model']:
    model = torch.load(f'models/ldm-basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)-10-1-0.pkl', map_location=device).to(device)
    if args['train_ddpm']:
        model.ddpm = DDPM(diffusion_depth=args['T'], latent_shape=model.enc_dec.latent_shape, label_dim=args['label_dim'], conv_map=conv_map).to(device)
else:
    model = LDM(diffusion_depth=args['T'], feature_channels=args['num_glyphs'], label_dim=args['label_dim'], conv_map=conv_map).to(device)
    # model = DDPM(diffusion_depth=args['T'], latent_shape=(1, 128*6, 128*5), label_dim=args['label_dim'], conv_map=conv_map).to(device)

mse_loss = args['loss_fn']
vae_optimizer = torch.optim.AdamW(model.enc_dec.parameters(), lr=args['vae_lr'], weight_decay=args['vae_weight_decay'])
ddpm_optimizer = torch.optim.AdamW(model.ddpm.parameters(), lr=args['ddpm_lr'], weight_decay=args['ddpm_weight_decay'])

max_len = 33928
num_glyphs = 26
step_every = 1
train_start, train_end = 0, int(0.95 * max_len) * num_glyphs
test_start, test_end = train_end, max_len * num_glyphs

im_dataset_name = "basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)"
im_dataset = torch.load(f'./{im_dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
im_dataset_test = torch.load(f'./{im_dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
vae_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
vae_train_dataloader = DataLoader(vae_train_tensor_dataset, batch_size=args['vae_batch_size'], shuffle=False)
vae_test_tensor_dataset = TensorDataset(im_dataset_test, torch.zeros(im_dataset_test.shape[0], 1))
vae_test_dataloader = DataLoader(vae_test_tensor_dataset, batch_size=args['vae_batch_size'], shuffle=False)
ddpm_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
ddpm_train_dataloader = DataLoader(ddpm_train_tensor_dataset, batch_size=args['ddpm_batch_size'], shuffle=False)

if args['use_wandb']:
    wandb.init(
        project="project-typeface",
        config={
            "model_type": "Diffusion",
            **args
        }
    )

@torch.no_grad()
def sampling_traj(ldm, z_T, T, times, y, num_samples):
    i = T
    prior_eval = ldm.ddpm.training
    ldm.ddpm.eval()
    x_Ts = [ldm.enc_dec.decode(ldm.denormalize_z(z_T))]
    # x_Ts = [z_T]
    while i >= 1:
        z_T = ldm.ddpm.denoise(z_T, times[i:i+1], y)
        i -= 1
        if i % (T // (num_samples - 1)) == 0:
            x_Ts.append(ldm.enc_dec.decode(ldm.denormalize_z(z_T)))
            # x_Ts.append(z_T)
    ldm.ddpm.train(prior_eval)
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

def recon_loss(a, b):
    return torch.pow((a - b), 2).sum()

def kl_loss(mu, logvar):
    return 0.5 * ((torch.pow(mu, 2) + logvar.exp() - logvar - 1)).sum()

if args["train_vae"]:
    model.enc_dec.train()
    model.ddpm.eval()
    for epoch in range(args['vae_epochs']):
        total_loss = 0
        test_loss = 0
        model.enc_dec.train()
        for inp, label in tqdm(vae_train_dataloader):
            vae_optimizer.zero_grad()
            inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
            inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
            inp_hat, mu, logvar = model.enc_dec(inp)
            loss = (recon_loss(inp_hat, inp) + args['vae_beta'] * kl_loss(mu, logvar)) / inp.shape[0]
            total_loss += loss.item() * inp.shape[0]
            loss.backward()
            if args['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['gradient_clip_val'])
            vae_optimizer.step()
            torch.cuda.empty_cache()
        
        model.enc_dec.eval()
        with torch.no_grad():
            for inp, label in tqdm(vae_test_dataloader):
                inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
                inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
                inp_hat, mu, logvar = model.enc_dec(inp)
                loss = (recon_loss(inp_hat, inp) + args['vae_beta'] * kl_loss(mu, logvar)) / inp.shape[0]
                test_loss += loss.item() * inp.shape[0]
                torch.cuda.empty_cache()
        total_loss /= len(vae_train_dataloader.dataset)
        test_loss /= len(vae_test_dataloader.dataset)

        test_idx = np.random.randint(0, im_dataset_test.shape[0] // args["num_glyphs"])
        inp, _ = vae_test_dataloader.dataset[test_idx*args["num_glyphs"]:(test_idx + 1)*args["num_glyphs"]]
        inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
        inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
        sample, mu, logvar = model.enc_dec(inp)
        two_five_five_truth = ((inp[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
        two_five_five_sample = ((sample[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()

        wandb.log({
            "vae_train_loss": total_loss,
            "vae_test_loss": test_loss,
            "vae_truth": wandb.Image(two_five_five_truth, caption=f"epoch{epoch+1}.png"),
            "vae_recon": wandb.Image(two_five_five_sample, caption=f"epoch{epoch+1}.png"),
        })
        print(f"Epoch {epoch+1}/{args['vae_epochs']}: {total_loss=}, {test_loss=}, {mu.abs().mean().item()=}, {logvar.abs().mean().item()=}")
    torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vae_beta"]).split("."))}-{args["vae_epochs"]}-0.pkl')

if args['rescale_latent']:
    print("rescaling latent space")
    z = torch.zeros(vae_train_tensor_dataset.tensors[0].shape[0], model.enc_dec.latent_shape[0], model.enc_dec.latent_shape[1], model.enc_dec.latent_shape[2], dtype=args['precision'])
    c = 0
    for inp, y in tqdm(vae_train_dataloader):
        inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
        inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
        z[c:c+inp.shape[0]] = (model.enc_dec.encode(inp)).cpu().detach()
        c += inp.shape[0]
        torch.cuda.empty_cache()

    min_z = torch.Tensor([[[z[:,i,j,k].min() for k in range(model.enc_dec.latent_shape[2])] for j in range(model.enc_dec.latent_shape[1])] for i in range(model.enc_dec.latent_shape[0])]).to(device)
    max_z = torch.Tensor([[[z[:,i,j,k].max() for k in range(model.enc_dec.latent_shape[2])] for j in range(model.enc_dec.latent_shape[1])] for i in range(model.enc_dec.latent_shape[0])]).to(device)
    model.set_latent_range(min_z.min(), max_z.max())
    print(min_z.mean(), max_z.mean())
else:
    model.set_latent_range(-torch.ones(model.enc_dec.latent_shape).to(device), torch.ones(model.enc_dec.latent_shape).to(device))

model.enc_dec.eval()
if args['train_ddpm']:
    for epoch in range(args['ddpm_epochs']):
        total_loss = 0
        model.ddpm.train()
        for inp, label in tqdm(ddpm_train_dataloader):
            ddpm_optimizer.zero_grad()

            # # inp (bs * 26, 1, 128, 128)
            # inp = inp.reshape(inp.shape[0] // 26, 26, 1, 128, 128)
            # r1 = torch.cat(inp[:,0:5].chunk(5, dim=1), dim=-1)
            # r2 = torch.cat(inp[:,5:10].chunk(5, dim=1), dim=-1)
            # r3 = torch.cat(inp[:,10:15].chunk(5, dim=1), dim=-1)
            # r4 = torch.cat(inp[:,15:20].chunk(5, dim=1), dim=-1)
            # r5 = torch.cat(inp[:,20:25].chunk(5, dim=1), dim=-1)
            # r6 = torch.cat([inp[:,25:26], torch.zeros(inp[:,25:26].shape[0], 1, 128 * 4, 128).to(device)], dim=-1)
            # inp = torch.cat((r1, r2, r3, r4, r5, r6), dim=-2)[:,0] # (bs, 1, 128*6, 128*5)

            inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
            times_i = torch.randint(1, args['T']+1, (inp.shape[0],)).to(device)
            inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
            label = label.to(device)
            if np.random.random() < 0.1:
                label = None
            label = None

            eps, pred_eps = model(inp, times_i, label)
            loss = mse_loss(pred_eps, eps)
            total_loss += loss.item() * inp.shape[0]
            loss.backward()
            ddpm_optimizer.step()

            torch.cuda.empty_cache()
        total_loss /= len(ddpm_train_dataloader.dataset)
        
        if True:
            num_images = 9
            # plt.show()
            shape = (1, 64, 16, 16)
            noise = model.enc_dec.reparameterize(torch.zeros(shape).to(device), torch.ones(shape).to(device))[0]
            times = torch.IntTensor(np.linspace(0, args['T'], args['T']+1, dtype=int)).to(device)
            condition = None#torch.Tensor([[np.random.randint(1)]]).to(device)
            traj, cond = sampling_traj(model, noise, args['T'], times, condition, num_images)
            for i in range(num_images):
                plt.subplot(1, num_images+1, i+1)
                show_img_from_tensor(traj[i][0,0:1])
            plt.show()
            log_images = {"train_loss": total_loss}
            # for i in range(args['num_glyphs']):
            #     two_five_five = ((traj[-1][0,i:i+1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
            #     log_images[f"images_{i}"] = wandb.Image(two_five_five, caption=f"epoch{epoch+1}.png")

            base_img = (128, 128)
            out_img = torch.ones(1, base_img[0]*6, base_img[1]*5) * 255.0
            for i in range(args['num_glyphs']):
                r = i // 5
                c = i % 5
                out_img[:,r*base_img[0]:(r+1)*base_img[0],c*base_img[1]:(c+1)*base_img[1]] = ((traj[-1][0,i:i+1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
            log_images["all_glyphs"] = wandb.Image(out_img, caption=f"epoch{epoch+1}.png")

            if args['use_wandb']:
                wandb.log(log_images)

        if cond:
            print(f"Condition: {cond.cpu().detach().numpy()[0]}\n{total_loss=}\nEpoch {epoch+1}/{args['ddpm_epochs']} finished.")
        else:
            print(f"{total_loss=}\nEpoch {epoch+1}/{args['ddpm_epochs']} finished.")

        if (epoch+1) % 100 == 0:
            torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vae_beta"]).split("."))}-{args["vae_epochs"]}-{epoch+1}.pkl')

    torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vae_beta"]).split("."))}-{args["vae_epochs"]}-{args["ddpm_epochs"]}.pkl')
