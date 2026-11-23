import os
import sys, time, torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase5_train_GAN import UNet3D_Generator
from exp1_train_gan import Generator3D

DEV = torch.device("cuda")

def time_mask_model(ckpt_path, patch_size, n=30):
    g = UNet3D_Generator().to(DEV)
    g.load_state_dict(torch.load(ckpt_path, map_location=DEV))
    g.eval()
    mask = torch.rand(1,1,patch_size,patch_size,patch_size, device=DEV)
    with torch.no_grad():
        _ = g(mask)
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            torch.cuda.synchronize(); t0=time.time()
            _ = g(mask)
            torch.cuda.synchronize()
            times.append(time.time()-t0)
    return np.mean(times), np.std(times)

def time_latent_model(ckpt_path, patch_size, latent_dim=128, n=30):
    g = Generator3D(latent_dim=latent_dim, output_size=patch_size).to(DEV)
    g.load_state_dict(torch.load(ckpt_path, map_location=DEV))
    g.eval()
    z = torch.randn(1, latent_dim, device=DEV)
    with torch.no_grad():
        _ = g(z)
        torch.cuda.synchronize()
        times = []
        for _ in range(n):
            torch.cuda.synchronize(); t0=time.time()
            _ = g(z)
            torch.cuda.synchronize()
            times.append(time.time()-t0)
    return np.mean(times), np.std(times)

configs = [
    ("TITAN-GAN", "mask", 32, "./output/models/tighterTitan/cgan_gen_epoch_350.pth"),
    ("Vanilla cGAN", "mask", 32, "./output/exp2_vanilla_cgan/vanilla_gen_epoch_300.pth"),
    ("WGAN-GP", "latent", 32, "./output/models/baseline_gan/baseline_gen_epoch_050.pth"),
    ("alpha-WGAN", "latent", 32, "./output/models/alpha_wgan/alpha_wgan_gen_epoch_300.pth"),
    ("TITAN-GAN", "mask", 64, "./output/res64/models/tighterTitan/cgan_gen_epoch_400.pth"),
    ("Vanilla cGAN", "mask", 64, "./output/res64/exp2_vanilla_cgan/vanilla_gen_epoch_350.pth"),
    ("WGAN-GP", "latent", 64, "./output/res64/models/baseline_gan/baseline_gen_epoch_050.pth"),
    ("alpha-WGAN", "latent", 64, "./output/res64/models/alpha_wgan/alpha_wgan_gen_epoch_050.pth"),
]

print(f"{'Model':<15}{'Res':<6}{'Mean (ms)':<12}{'Std (ms)':<10}")
for name, kind, res, ckpt in configs:
    fn = time_mask_model if kind=="mask" else time_latent_model
    mean, std = fn(ckpt, res)
    print(f"{name:<15}{res:<6}{mean*1000:<12.3f}{std*1000:<10.3f}")
