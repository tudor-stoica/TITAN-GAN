import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

# Paired Dataset Loader (identical convention to phase5_train_GAN.py / exp2_vanilla_cgan.py)
class PairedCTDataset(Dataset):
    def __init__(self, data_dir):
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, "*_mask.npy")))
        self.ct_files = [f.replace('_mask.npy', '_ct.npy') for f in self.mask_files]

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask = np.load(self.mask_files[idx])
        ct = np.load(self.ct_files[idx])

        if np.random.rand() > 0.5:
            mask, ct = np.flip(mask, axis=0), np.flip(ct, axis=0)
        if np.random.rand() > 0.5:
            mask, ct = np.flip(mask, axis=1), np.flip(ct, axis=1)
        if np.random.rand() > 0.5:
            mask, ct = np.flip(mask, axis=2), np.flip(ct, axis=2)
        k_rot = np.random.randint(0, 4)
        if k_rot > 0:
            mask, ct = np.rot90(mask, k=k_rot, axes=(0, 1)), np.rot90(ct, k=k_rot, axes=(0, 1))

        return torch.from_numpy(mask.copy()).unsqueeze(0).float(), torch.from_numpy(ct.copy()).unsqueeze(0).float()

def setup_args():
    parser = argparse.ArgumentParser(description="Exp 7: Train mask-conditioned 3D DDPM (MONAI DiffusionModelUNet)")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--out-dir", type=str, default="./output/exp7_diffusion")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-train-timesteps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing mask-conditioned 3D DDPM on: {device}")

    dataset = PairedCTDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print(f"Training data: {len(dataset)} paired (mask, CT) patches, {len(dataloader)} steps/epoch")

    # Mask-conditioned via channel concat: in_channels = noisy_ct(1) + mask(1) = 2, out_channels = predicted noise on the CT channel (1)
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 128),
        num_res_blocks=1,
        attention_levels=(False, False, False, True),
        num_head_channels=32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps, schedule="linear_beta", prediction_type="epsilon")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history = []
    print("Starting diffusion training loop...")
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for real_mask, real_ct in dataloader:
            real_mask = real_mask.to(device)
            real_ct = real_ct.to(device)

            # Rescale CT from [0,1] (Phase 1 normalization) to [-1,1] to match scheduler's default clip range.
            # Also rescale the mask to [-1,1] so the conditioning channel has the same dynamic range/energy
            # as the noisy CT channel - a {0,1} mask concatenated next to a N(0,1)-ish noisy channel is a
            # comparatively weak signal for the network's early conv filters to pick up on.
            real_ct_scaled = real_ct * 2.0 - 1.0
            real_mask_scaled = real_mask * 2.0 - 1.0

            noise = torch.randn_like(real_ct_scaled)
            timesteps = torch.randint(0, args.num_train_timesteps, (real_ct_scaled.shape[0],), device=device).long()

            noisy_ct = scheduler.add_noise(original_samples=real_ct_scaled, noise=noise, timesteps=timesteps)
            model_input = torch.cat([noisy_ct, real_mask_scaled], dim=1)

            optimizer.zero_grad()
            pred_noise = model(x=model_input, timesteps=timesteps)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        history.append(avg_loss)

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - training_start
            print(f"Epoch [{epoch:03d}/{args.epochs}] | MSE Loss: {avg_loss:.5f} | Epoch time: {epoch_time:.2f}s | Elapsed: {elapsed/60:.1f}min")

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f"diffusion_unet_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> [CHECKPOINT SAVED] {ckpt_path}")

    total_training_time = time.time() - training_start
    print(f"\nTotal training time: {total_training_time:.1f}s ({total_training_time/60:.2f} min) for {args.epochs} epochs")

    with open(os.path.join(args.out_dir, "training_time.txt"), "w") as f:
        f.write(f"total_training_seconds={total_training_time:.2f}\n")
        f.write(f"total_training_minutes={total_training_time/60:.3f}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"n_train_samples={len(dataset)}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"num_train_timesteps={args.num_train_timesteps}\n")
        f.write(f"n_params={n_params}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), history, color="darkred", linewidth=2)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("MSE Denoising Loss", fontweight="bold")
    plt.title("Mask-Conditioned 3D DDPM Training", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "diffusion_training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
