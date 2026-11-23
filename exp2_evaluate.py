import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# ==========================================
# 1. Generator Architecture (Vanilla)
# ==========================================
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

# ==========================================
# 2. Evaluation & Minting Logic
# ==========================================
def setup_args():
    parser = argparse.ArgumentParser(description="Exp 2: Evaluate Vanilla cGAN")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data", help="Real Validation Data")
    parser.add_argument("--mask-dir", type=str, default="./output/phase4_synthetic_data", help="Phase 4 VAE Masks")
    parser.add_argument("--models-dir", type=str, default="./output/exp2_vanilla_cgan", help="Vanilla checkpoints")
    parser.add_argument("--out-dir", type=str, default="./output/exp2_vanilla_cgan/synthetic_data")
    parser.add_argument("--num-eval", type=int, default=100, help="Number of real pairs to evaluate against")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(glob.glob(os.path.join(args.models_dir, "vanilla_gen_epoch_*.pth")))
    if not checkpoints:
        print("Error: No vanilla checkpoints found.")
        return

    # Load Real Data for SSIM evaluation
    real_mask_files = sorted(glob.glob(os.path.join(args.data_dir, "*_mask.npy")))[:args.num_eval]
    real_ct_files = [f.replace('_mask.npy', '_ct.npy') for f in real_mask_files]

    generator = UNet3D_Generator().to(device)
    results = []

    print(f"Evaluating {len(checkpoints)} Vanilla Checkpoints...")

    # --- Step 1: Find the Best Epoch ---
    for ckpt_path in checkpoints:
        epoch_num = int(ckpt_path.split('_')[-1].split('.')[0])
        generator.load_state_dict(torch.load(ckpt_path, map_location=device))
        generator.eval()

        epoch_ssim, epoch_psnr = 0.0, 0.0

        with torch.no_grad():
            for i in tqdm(range(len(real_mask_files)), desc=f"Epoch {epoch_num}", leave=False):
                real_mask_np = np.load(real_mask_files[i])
                real_ct_np = np.load(real_ct_files[i])

                mask_tensor = torch.from_numpy(real_mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
                fake_ct_np = generator(mask_tensor).squeeze().cpu().numpy()

                epoch_ssim += ssim(real_ct_np, fake_ct_np, data_range=1.0)
                epoch_psnr += psnr(real_ct_np, fake_ct_np, data_range=1.0)

        avg_ssim = epoch_ssim / len(real_mask_files)
        avg_psnr = epoch_psnr / len(real_mask_files)
        results.append({'Epoch': epoch_num, 'SSIM': avg_ssim, 'PSNR': avg_psnr})
        print(f"Epoch {epoch_num} -> SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f}dB")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.models_dir, "vanilla_evaluation.csv"), index=False)
    
    best_epoch = int(df.loc[df['SSIM'].idxmax()]['Epoch'])
    print(f"\nEvaluation Complete! Best Vanilla Epoch is {best_epoch}.")

    # Mint 600 Final Patches from VAE Masks
    print(f"\nLoading Phase 4 Synthetic Masks to mint final Vanilla CT patches...")
    best_ckpt_path = os.path.join(args.models_dir, f"vanilla_gen_epoch_{best_epoch:03d}.pth")
    generator.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    generator.eval()

    synth_mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*_mask.npy")))

    with torch.no_grad():
        for mask_path in tqdm(synth_mask_files, desc="Minting Vanilla CTs"):
            mask_np = np.load(mask_path)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
            fake_ct_np = generator(mask_tensor).squeeze().cpu().numpy()
            
            base_name = os.path.basename(mask_path).replace('_mask.npy', '')
            out_path = os.path.join(args.out_dir, f"vanilla_ct_{base_name}.npy")
            np.save(out_path, fake_ct_np)

    print(f"\nSuccess! Vanilla ablation data secured at {args.out_dir}")

if __name__ == "__main__":
    main()