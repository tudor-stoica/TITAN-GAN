import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler


def build_diffusion_model(device):
    """Same model definition as exp7_evaluate_diffusion.py."""
    return DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 128), num_res_blocks=1,
        attention_levels=(False, False, False, True), num_head_channels=32,
    ).to(device)


@torch.no_grad()
def ddim_sample(model, mask_tensor, ddim_scheduler, device, num_inference_steps=40):
    """Same DDIM sampler as exp7_evaluate_diffusion.py."""
    ddim_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    mask_scaled = mask_tensor * 2.0 - 1.0
    x = torch.randn_like(mask_tensor, device=device)
    for t in ddim_scheduler.timesteps:
        model_input = torch.cat([x, mask_scaled], dim=1)
        t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        pred_noise = model(x=model_input, timesteps=t_batch)
        x, _ = ddim_scheduler.step(pred_noise, t, x)
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


# The Generators
# The U-Net
class UNet3D_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        self.down1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._block(features * 4, features * 8)
        self.up1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.up_conv1 = self._block(features * 8, features * 4)
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.up_conv2 = self._block(features * 4, features * 2)
        self.up3 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.up_conv3 = self._block(features * 2, features)
        self.final = nn.Sequential(
            nn.Conv3d(features, out_channels, kernel_size=1),
            nn.Sigmoid() 
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))
        u1 = self.up_conv1(torch.cat([self.up1(b), d3], dim=1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), d2], dim=1))
        u3 = self.up_conv3(torch.cat([self.up3(u2), d1], dim=1))
        return self.final(u3)

# The Baseline WGAN-GP
class Generator3D(nn.Module):
    """Same architecture as Generator3D in exp1_train_gan.py."""
    def __init__(self, latent_dim=128, features=64, output_size=32):
        super().__init__()
        self.init_size = 4
        num_stages = round(np.log2(output_size / self.init_size))
        assert self.init_size * (2 ** num_stages) == output_size, \
            f"output_size={output_size} must be init_size({self.init_size}) * a power of 2"

        channel_list = [features * 4]
        for _ in range(num_stages - 1):
            channel_list.append(max(features, channel_list[-1] // 2))

        self.l1 = nn.Sequential(nn.Linear(latent_dim, channel_list[0] * self.init_size**3))

        layers = [nn.BatchNorm3d(channel_list[0])]
        for i in range(num_stages):
            in_ch = channel_list[i]
            is_last = (i == num_stages - 1)
            out_ch = 1 if is_last else channel_list[i + 1]
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            if is_last:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.BatchNorm3d(out_ch, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_blocks = nn.Sequential(*layers)
        self.first_channels = channel_list[0]

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.first_channels, self.init_size, self.init_size, self.init_size)
        return self.conv_blocks(out)

# Paired Dataset Loader
class PairedCTDataset(Dataset):
    def __init__(self, data_dir):
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, "*_mask.npy")))
        self.ct_files = [f.replace('_mask.npy', '_ct.npy') for f in self.mask_files]
        
    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask = np.load(self.mask_files[idx])
        ct = np.load(self.ct_files[idx])
        return torch.from_numpy(mask.copy()).unsqueeze(0).float(), torch.from_numpy(ct.copy()).unsqueeze(0).float()

# Main Logic
def setup_args():
    parser = argparse.ArgumentParser(description="Qualitative Model Comparison Visualizer")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--baseline-path", type=str, default="./output/models/baseline_gan/baseline_gen_epoch_050.pth")
    parser.add_argument("--vanilla-path", type=str, default="./output/exp2_vanilla_cgan/vanilla_gen_epoch_300.pth")
    parser.add_argument("--cascading-path", type=str, default="./output/models/tighterTitan/cgan_gen_epoch_350.pth")
    parser.add_argument("--alpha-wgan-path", type=str, default=None, help="Optional: alpha-WGAN generator checkpoint")
    parser.add_argument("--diffusion-path", type=str, default=None, help="Optional: mask-conditioned DDPM checkpoint")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="DDIM sampling steps (only used if --diffusion-path given)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-path", type=str, default="model_comparison_qualitative_new.png")
    parser.add_argument("--patch-size", type=int, default=32, help="Cubic patch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection (reproducible variants)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated explicit dataset indices to use instead of random selection, e.g. '316,549,296'")
    parser.add_argument("--no-labels", action="store_true", help="Omit the 'Sample N' row labels from the figure")
    return parser.parse_args()

def main():
    args = setup_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading Models onto: {device}")

    # Initialize models with correct architectures
    unconditional_models = {'Baseline'}
    models = {
        'Baseline': Generator3D(latent_dim=128, output_size=args.patch_size).to(device),
        'Vanilla': UNet3D_Generator().to(device),
        'Dual-Loss': UNet3D_Generator().to(device)
    }
    paths = {
        'Baseline': args.baseline_path,
        'Vanilla': args.vanilla_path,
        'Dual-Loss': args.cascading_path
    }
    if args.alpha_wgan_path:
        unconditional_models.add('Alpha-WGAN')
        models['Alpha-WGAN'] = Generator3D(latent_dim=128, output_size=args.patch_size).to(device)
        paths['Alpha-WGAN'] = args.alpha_wgan_path

    diffusion_model = None
    ddim_scheduler = None
    if args.diffusion_path:
        diffusion_model = build_diffusion_model(device)
        if os.path.exists(args.diffusion_path):
            diffusion_model.load_state_dict(torch.load(args.diffusion_path, map_location=device))
            diffusion_model.eval()
            ddim_scheduler = DDIMScheduler(num_train_timesteps=500, schedule="linear_beta",
                                            beta_start=0.0015, beta_end=0.0195, clip_sample=True)
            print("Loaded Diffusion model.")
        else:
            print(f"Error: Diffusion model weights not found at {args.diffusion_path}!")
            return

    for name, model in models.items():
        if os.path.exists(paths[name]):
            model.load_state_dict(torch.load(paths[name], map_location=device))
            model.eval()
            print(f"Loaded {name} model.")
        else:
            print(f"Error: {name} model weights not found at {paths[name]}!")
            return # Stop the script if weights are missing
            
    dataset = PairedCTDataset(args.data_dir)
    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        rng = np.random.RandomState(args.seed) if args.seed is not None else np.random
        indices = rng.choice(len(dataset), 5, replace=False)

    log_path = os.path.splitext(args.out_path)[0] + "_samples.txt"
    with open(log_path, "w") as f:
        for row, idx in enumerate(indices):
            mask_name = os.path.basename(dataset.mask_files[idx])
            f.write(f"row {row+1}: dataset_idx={idx}  file={mask_name}\n")
    print(f"Wrote sample log to {log_path}")

    n_cols = 2 + len(models) + (1 if diffusion_model is not None else 0)
    n_rows = len(indices)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axs = axs[None, :]
    col_titles = {'Baseline': 'WGAN-GP', 'Vanilla': 'Vanilla-TITAN',
                  'Dual-Loss': 'TITAN-GAN', 'Alpha-WGAN': 'alpha-WGAN'}
    cols = ['Input Mask', 'Ground Truth CT'] + [col_titles[name] for name in models.keys()]
    if diffusion_model is not None:
        cols.append('Diffusion')
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontweight='bold', fontsize=12)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            real_mask_tensor, real_ct_tensor = dataset[idx]
            
            mask_np = real_mask_tensor.squeeze().cpu().numpy()
            ct_np = real_ct_tensor.squeeze().cpu().numpy()
            mid_z = mask_np.shape[0] // 2
            
            # Input Mask & GT
            axs[row, 0].imshow(mask_np[mid_z, :, :], cmap='gray')
            if not args.no_labels:
                axs[row, 0].set_ylabel(f'Sample {idx}', fontweight='bold', rotation=0, labelpad=20)
            axs[row, 0].set_xticks([]); axs[row, 0].set_yticks([])
            axs[row, 1].imshow(ct_np[mid_z, :, :], cmap='bone', vmin=0, vmax=1)
            axs[row, 1].set_xticks([]); axs[row, 1].set_yticks([])

            mask_input = real_mask_tensor.unsqueeze(0).to(device)

            col_offset = 2
            for name, model in models.items():
                if name in unconditional_models:
                    # Unconditional models take random noise, not the mask
                    z = torch.randn(1, 128).to(device)
                    fake_ct_tensor = model(z)
                else:
                    # U-Nets take the mask
                    fake_ct_tensor = model(mask_input)
                    
                fake_ct_np = fake_ct_tensor.squeeze().cpu().numpy()
                axs[row, col_offset].imshow(fake_ct_np[mid_z, :, :], cmap='bone', vmin=0, vmax=1)
                axs[row, col_offset].set_xticks([]); axs[row, col_offset].set_yticks([])
                col_offset += 1

            if diffusion_model is not None:
                diff_ct_tensor = ddim_sample(diffusion_model, mask_input, ddim_scheduler, device, args.num_inference_steps)
                diff_ct_np = diff_ct_tensor.squeeze().cpu().numpy()
                axs[row, col_offset].imshow(diff_ct_np[mid_z, :, :], cmap='bone', vmin=0, vmax=1)
                axs[row, col_offset].set_xticks([]); axs[row, col_offset].set_yticks([])
                col_offset += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(args.out_path, dpi=150)
    print(f"Visualization secured at {args.out_path}. Qualitative comparison complete.")

if __name__ == "__main__":
    main()