import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, dataset
from ema_pytorch import EMA
import numpy as np
from backend.ml.ddpm import DDPM
from backend.ml.ldm import LDM
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pprint import pprint
from backend.config import conv_map, device

print(f"Executing train-diffusion.ipynb on {device}...\n-----------------------------")

args = {
    "load_model": True,
    "model_load_string": 'models/ldm-basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)-0005-100-0.pkl',
    "train_vp": False,
    "train_ddpm": True,
    "use_wandb": True,
    "vp_epochs": 100,
    "ddpm_epochs": 2500,
    "vp_batch_size": 26 * 16,
    "ddpm_batch_size": 26 * 64,
    "vp_lr": 4e-4,
    "ddpm_lr": 1e-4,
    "vp_weight_decay": 1e-5,
    "ddpm_weight_decay": 1e-5,
    "vp_beta": 5e-3,
    "gradient_clip": False,
    "gradient_clip_val": 1.0,
    "T": 1024,
    "num_glyphs": 26,
    "embedding_dim": 2048,
    "num_layers": 24,
    "num_heads": 32,
    "label_dim": 128,
    "sample_every": 50,
    "loss_fn": nn.MSELoss(),
    "precision": torch.float32,
    "rescale_latent": False,
    "save_after_scaling": True,
    "vp_use_scheduler": True,
    "vp_scheduler_warmup_steps": 2000,
    "ddpm_use_scheduler": False,
    "ddpm_scheduler_warmup_steps": 2000,
}

print("Training hyperparameters:")
pprint(args)

if args['load_model']:
    model = torch.load(args['model_load_string'], map_location=device, weights_only=False).to(device, dtype=args['precision'])
    model.ddpm = DDPM(diffusion_depth=args['T'], num_layers=args['num_layers'], embedding_dim=args['embedding_dim'], num_glyphs=args['num_glyphs'], num_heads=args['num_heads'], cond_dim=args['label_dim']).to(device, dtype=args['precision'])
else:
    # model = DDPM(diffusion_depth=args['T'], latent_shape=(1, 128*6, 128*5), label_dim=args['label_dim'], conv_map=conv_map).to(device, dtype=args['precision'])
    model = LDM(diffusion_depth=args['T'], embedding_dim=args['embedding_dim'], num_glyphs=args['num_glyphs'], label_dim=args['label_dim'], num_layers=args['num_layers'], num_heads=args['num_heads'], cond_dim=args['label_dim']).to(device, dtype=args['precision'])

print(torch.cuda.memory_summary(device=device, abbreviated=True))
ema = EMA(
    model,
    beta=0.9999,
    update_after_step=5,
    update_every=1
).to(dtype=args['precision'])

mse_loss = args['loss_fn']
vp_params = [param for param in model.enc_dec.parameters() if param.requires_grad]
vp_optimizer = torch.optim.AdamW(vp_params, lr=args['vp_lr'], weight_decay=args['vp_weight_decay'])

ddpm_params = [param for param in model.ddpm.parameters() if param.requires_grad]
ddpm_optimizer = torch.optim.AdamW(ddpm_params, lr=args['ddpm_lr'], weight_decay=args['ddpm_weight_decay'])

max_len = 33928
num_glyphs = 26
step_every = 1
train_start, train_end = 0, int(0.95 * max_len) * num_glyphs
test_start, test_end = train_end, max_len * num_glyphs

if args['vp_use_scheduler']:
    vp_batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['vp_batch_size'] + 0.5)
    vp_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(vp_optimizer, T_max=args['vp_epochs'] * vp_batches_per_epoch, eta_min=1e-5)
    vp_scheduler2 = torch.optim.lr_scheduler.LinearLR(vp_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['vp_scheduler_warmup_steps'])
    vp_scheduler = torch.optim.lr_scheduler.ChainedScheduler([vp_scheduler1, vp_scheduler2], optimizer=vp_optimizer)

if args['ddpm_use_scheduler']:
    ddpm_batches_per_epoch = int(max_len * (num_glyphs // step_every) / args['ddpm_batch_size'] + 0.5)
    ddpm_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(ddpm_optimizer, T_max=args['ddpm_epochs'] * ddpm_batches_per_epoch, eta_min=1e-5)
    ddpm_scheduler2 = torch.optim.lr_scheduler.LinearLR(ddpm_optimizer, start_factor=0.001, end_factor=1.0, total_iters=args['ddpm_scheduler_warmup_steps'])
    ddpm_scheduler = torch.optim.lr_scheduler.ChainedScheduler([ddpm_scheduler1, ddpm_scheduler2], optimizer=ddpm_optimizer)

im_dataset_name = "basic-33928allchars_centered_scaled_sorted_filtered_(128, 128)"
im_dataset = torch.load(f'./{im_dataset_name}.pt', mmap=True)[train_start:train_end:step_every]
im_dataset_test = torch.load(f'./{im_dataset_name}.pt', mmap=True)[test_start:test_end:step_every]
vp_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
vp_train_dataloader = DataLoader(vp_train_tensor_dataset, batch_size=args['vp_batch_size'], shuffle=False)
vp_test_tensor_dataset = TensorDataset(im_dataset_test, torch.zeros(im_dataset_test.shape[0], 1))
vp_test_dataloader = DataLoader(vp_test_tensor_dataset, batch_size=args['vp_batch_size'], shuffle=False)
ddpm_train_tensor_dataset = TensorDataset(im_dataset, torch.zeros(im_dataset.shape[0], 1))
ddpm_train_dataloader = DataLoader(ddpm_train_tensor_dataset, batch_size=args['ddpm_batch_size'], shuffle=False)

if args['use_wandb']:
    wandb.init(
        project="project-typeface",
        config={
            "model_type": "Diffusion-Transformer",
            **args
        }
    )

@torch.no_grad()
def sampling_traj(ldm, z_T, T, times, y, num_samples):
    i = T
    prior_eval = ldm.training
    ldm.eval()
    x_Ts = [ldm.latent_to_feature(z_T)]
    while i >= 1:
        z_T = ldm.denoise(z_T, times[i:i+1], y)
        i -= 1
        if i % (T // (num_samples - 1)) == 0:
            x_Ts.append(ldm.latent_to_feature(z_T))
    ldm.train(prior_eval)
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

if args["train_vp"]:
    for epoch in range(args['vp_epochs']):
        total_loss = 0
        test_loss = 0
        model.train()
        for step, (inp, label) in enumerate(tqdm(vp_train_dataloader)):
            vp_optimizer.zero_grad()
            inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
            inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
            inp_hat, mu, logvar = model.enc_dec(inp)
            loss = (recon_loss(inp_hat, inp) + kl_loss(mu, logvar)) / inp.shape[0]
            total_loss += loss.item() * inp.shape[0]
            loss.backward()
            if args['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['gradient_clip_val'])
            vp_optimizer.step()
            if args['vp_use_scheduler']:
                vp_scheduler.step()
            torch.cuda.empty_cache()

            if args['use_wandb']:
                if (step+1) % 100 == 0:
                    with torch.no_grad():
                        model.eval()
                        test_idx = np.random.randint(0, im_dataset_test.shape[0] // args["num_glyphs"])
                        inp, _ = vp_test_dataloader.dataset[test_idx*args["num_glyphs"]:(test_idx + 1)*args["num_glyphs"]]
                        inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
                        inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
                        sample, mu, logvar = model.enc_dec(inp)
                        two_five_five_truth = ((inp[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
                        two_five_five_sample = ((sample[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
                        model.train()

                        wandb.log({
                            "vp_truth": wandb.Image(two_five_five_truth, caption=f"epoch{epoch+1}_{step}.png"),
                            "vp_recon": wandb.Image(np.concatenate((two_five_five_truth.cpu().detach().numpy(), two_five_five_sample.cpu().detach().numpy()), axis=2), caption=f"epoch{epoch+1}_{step}.png"),
                            "mu_abs_mean": mu.abs().mean().item(),
                            "logvar_abs_mean": logvar.abs().mean().item(),
                            "vp_train_loss_step": loss.item(),
                            "vp_lr": vp_scheduler.get_last_lr()[0] if args['vp_use_scheduler'] else args['vp_lr']
                        })
                else:
                    wandb.log({
                        "vp_train_loss_step": loss.item(),
                        "mu_abs_mean": mu.abs().mean().item(),
                        "logvar_abs_mean": logvar.abs().mean().item(),
                        "vp_lr": vp_scheduler.get_last_lr()[0] if args['vp_use_scheduler'] else args['vp_lr']
                    })
        
        model.eval()
        with torch.no_grad():
            for inp, label in tqdm(vp_test_dataloader):
                inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
                inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
                inp_hat, mu, logvar = model.enc_dec(inp)
                loss = (recon_loss(inp_hat, inp) + kl_loss(mu, logvar)) / inp.shape[0]
                test_loss += loss.item() * inp.shape[0]
                torch.cuda.empty_cache()
        total_loss /= len(vp_train_dataloader.dataset)
        test_loss /= len(vp_test_dataloader.dataset)

        test_idx = np.random.randint(0, im_dataset_test.shape[0] // args["num_glyphs"])
        inp, _ = vp_test_dataloader.dataset[test_idx*args["num_glyphs"]:(test_idx + 1)*args["num_glyphs"]]
        inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
        inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
        sample, mu, logvar = model.enc_dec(inp)
        two_five_five_truth = ((inp[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
        two_five_five_sample = ((sample[0,:1] + 1.0) * 127.5).clamp(0.0, 255.0).round()

        wandb.log({
            "vp_test_loss": test_loss,
            "vp_truth": wandb.Image(two_five_five_truth, caption=f"epoch{epoch+1}.png"),
            "vp_recon": wandb.Image(np.concatenate((two_five_five_truth.cpu().detach().numpy(), two_five_five_sample.cpu().detach().numpy()), axis=2), caption=f"epoch{epoch+1}.png"),
        })
        print(f"Epoch {epoch+1}/{args['vp_epochs']}: {total_loss=}, {test_loss=}, {mu.abs().mean().item()=}, {logvar.abs().mean().item()=}")
    torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vp_beta"]).split("."))}-{args["vp_epochs"]}-0.pkl')

if args['rescale_latent']:
    print("rescaling latent space")
    min_z = torch.zeros(1, model.enc_dec.latent_shape[-2], model.enc_dec.latent_shape[-1], dtype=args['precision'])
    max_z = torch.zeros(1, model.enc_dec.latent_shape[-2], model.enc_dec.latent_shape[-1], dtype=args['precision'])
    for inp, y in tqdm(vp_train_dataloader):
        inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
        inp = inp.to(device, dtype=torch.uint8) / 127.5 - 1.0
        z = (model.enc_dec.encode(inp)).cpu().detach()
        min_z = torch.minimum(min_z, torch.min(z, dim=0).values)
        max_z = torch.maximum(max_z, torch.max(z, dim=0).values)
        torch.cuda.empty_cache()
    model.set_latent_range(min_z.to(device), max_z.to(device))
    print(min_z.mean(), max_z.mean())
    if args['save_after_scaling']:
        torch.save(model, args['model_load_string'])
else:
    pass
    # model.set_latent_range(-torch.ones(model.enc_dec.latent_shape).to(device), torch.ones(model.enc_dec.latent_shape).to(device))

def sample(epoch, step):
    num_images = 9
    # plt.show()
    # shape = (1, 1, 2048)
    shape = (1, args['num_glyphs'], 2048)
    # x = ddpm_train_dataloader.dataset[0:args['num_glyphs']][0].to(device, dtype=args['precision']).reshape(1, args['num_glyphs'], 128, 128) / 127.5 - 1.0
    # noise = model.noise_pred.embedder.vector_projector.encode(x)
    # recon = model.noise_pred.embedder.vector_projector.decode(noise)
    noise = model.ddpm.reparameterize(torch.zeros(shape).to(device, dtype=args['precision']), torch.ones(shape).to(device, dtype=args['precision']))[0]
    times = torch.IntTensor(np.linspace(0, args['T'], args['T']+1, dtype=int)).to(device)
    condition = None#torch.Tensor([[np.random.randint(1)]]).to(device)
    traj, cond = sampling_traj(model, noise, args['T'], times, condition, num_images)
    for i in range(num_images):
        plt.subplot(1, num_images+1, i+1)
        # show_img_from_tensor(traj[i][0,0:1])
    plt.show()
    # for i in range(args['num_glyphs']):
    #     two_five_five = ((traj[-1][0,i:i+1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
    #     log_images[f"images_{i}"] = wandb.Image(two_five_five, caption=f"epoch{epoch+1}.png")

    base_img = (128, 128)
    out_img = torch.ones(1, base_img[0]*6, base_img[1]*5) * 255.0
    # for i in range(1):
    for i in range(args['num_glyphs']):
        r = i // 5
        c = i % 5
        out_img[:,r*base_img[0]:(r+1)*base_img[0],c*base_img[1]:(c+1)*base_img[1]] = ((traj[-1][0,i:i+1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
        # out_img[:,r*base_img[0]:(r+1)*base_img[0],c*base_img[1]:(c+1)*base_img[1]] = ((recon[0,i:i+1] + 1.0) * 127.5).clamp(0.0, 255.0).round()
    log_images = {"all_glyphs": wandb.Image(out_img, caption=f"epoch{epoch+1}_{step}.png")}

    return log_images

if args['train_ddpm']:
    for epoch in range(args['ddpm_epochs']):
        total_loss = 0
        model.train()
        for step, (inp, label) in enumerate(tqdm(ddpm_train_dataloader)):
            ddpm_optimizer.zero_grad()

            inp = inp.reshape(inp.shape[0] // args["num_glyphs"], args["num_glyphs"], 128, 128)
            times_i = torch.randint(1, args['T']+1, (inp.shape[0],)).to(device)
            inp = inp.to(device, dtype=args['precision']) / 127.5 - 1.0
            label = label.to(device)
            if np.random.random() < 0.1:
                label = None
            label = None
            
            eps, pred_eps = model(inp, times_i, label)
            loss = recon_loss(pred_eps, eps) / inp.shape[0]
            total_loss += loss.item() * inp.shape[0]
            loss.backward()
            ddpm_optimizer.step()
            if args['ddpm_use_scheduler']:
                ddpm_scheduler.step()
            ema.update()

            torch.cuda.empty_cache()

            log_images = {
                "train_loss_step": loss.item(),
                "ddpm_lr": ddpm_scheduler.get_last_lr()[0] if args['ddpm_use_scheduler'] else args['ddpm_lr']
            }
            if (step+1) % args["sample_every"] == 0:
                log_images.update(sample(epoch, step))
            if args['use_wandb']:
                wandb.log(log_images)
        total_loss /= len(ddpm_train_dataloader.dataset)
        

        print(f"{total_loss=}\nEpoch {epoch+1}/{args['ddpm_epochs']} finished.")

        if (epoch+1) % 100 == 0:
            torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vp_beta"]).split("."))}-{args["vp_epochs"]}-{epoch+1}.pkl')

    torch.save(model, f'models/ldm-{im_dataset_name}-{"".join(str(args["vp_beta"]).split("."))}-{args["vp_epochs"]}-{args["ddpm_epochs"]}.pkl')
