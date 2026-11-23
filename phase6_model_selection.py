import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

# Architectures
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

class SoftGraphBridge(nn.Module):
    def __init__(self):
        super(SoftGraphBridge, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

# Evaluation Logic
def calculate_dice(preds, targets, smooth=1e-5):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def setup_args():
    parser = argparse.ArgumentParser(description="Phase 6: Evaluate cGAN Checkpoints")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data", help="Real validation pairs")
    parser.add_argument("--models-dir", type=str, default="./output/models/tighterTitan", help="Directory with GAN checkpoints")
    parser.add_argument("--phase2-path", type=str, default="./output/models/soft_graph_bridge_best.pth", help="Phase 2 CNN")
    parser.add_argument("--out-dir", type=str, default="./output/gan_evaluation_New")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of real pairs to evaluate against")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Phase 2 Topology Enforcer
    print("Loading Phase 2 Topology Enforcer...")
    topo_net = SoftGraphBridge().to(device)
    topo_net.load_state_dict(torch.load(args.phase2_path, map_location=device))
    topo_net.eval()

    # Find all Generator checkpoints
    checkpoints = sorted(glob.glob(os.path.join(args.models_dir, "cgan_gen_epoch_*.pth")))
    if not checkpoints:
        print("Error: No generator checkpoints found.")
        return

    # Load Validation Data
    mask_files = sorted(glob.glob(os.path.join(args.data_dir, "*_mask.npy")))[:args.num_samples]
    ct_files = [f.replace('_mask.npy', '_ct.npy') for f in mask_files]
    
    results = []

    print(f"Evaluating {len(checkpoints)} checkpoints across {len(mask_files)} validation samples...")

    generator = UNet3D_Generator().to(device)

    # Loop through each epoch checkpoint
    for ckpt_path in checkpoints:
        epoch_num = int(ckpt_path.split('_')[-1].split('.')[0])
        print(f"\n--- Evaluating Epoch {epoch_num} ---")
        
        generator.load_state_dict(torch.load(ckpt_path, map_location=device))
        generator.eval()

        epoch_ssim, epoch_psnr, epoch_topo_dice = 0.0, 0.0, 0.0

        with torch.no_grad():
            for i in tqdm(range(len(mask_files)), desc="Calculating Metrics"):
                # Load real data
                real_mask_np = np.load(mask_files[i])
                real_ct_np = np.load(ct_files[i])

                mask_tensor = torch.from_numpy(real_mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
                real_mask_tensor = mask_tensor.clone()

                # Generate Fake CT
                fake_ct_tensor = generator(mask_tensor)
                fake_ct_np = fake_ct_tensor.squeeze().cpu().numpy()

                # Metrics: SSIM & PSNR (Data range is strictly 0 to 1)
                epoch_ssim += ssim(real_ct_np, fake_ct_np, data_range=1.0)
                epoch_psnr += psnr(real_ct_np, fake_ct_np, data_range=1.0)

                # Metrics: Topology Verification
                fake_mask_prediction = topo_net(fake_ct_tensor)
                epoch_topo_dice += calculate_dice(fake_mask_prediction, real_mask_tensor).item()

        # Average metrics for this epoch
        avg_ssim = epoch_ssim / len(mask_files)
        avg_psnr = epoch_psnr / len(mask_files)
        avg_topo_dice = epoch_topo_dice / len(mask_files)

        results.append({
            'Epoch': epoch_num,
            'SSIM': avg_ssim,
            'PSNR': avg_psnr,
            'Topo_Dice': avg_topo_dice
        })
        
        print(f"Results -> SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f}dB | Topo Dice: {avg_topo_dice:.4f}")

    # Save and Plot Results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "cgan_evaluation_metrics.csv"), index=False)

    print("\nGenerating Evaluation Plot...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(df['Epoch'], df['SSIM'], marker='o', color='blue', linewidth=2)
    axs[0].set_title('Structural Similarity Index (SSIM) - Higher is Better', fontweight='bold')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df['Epoch'], df['PSNR'], marker='o', color='green', linewidth=2)
    axs[1].set_title('Peak Signal-to-Noise Ratio (PSNR) - Higher is Better', fontweight='bold')
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(df['Epoch'], df['Topo_Dice'], marker='o', color='purple', linewidth=2)
    axs[2].set_title('Topological Preservation (Dice) - Higher is Better', fontweight='bold')
    axs[2].set_xlabel('Epochs', fontweight='bold')
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cgan_evaluation_plot.png"), dpi=150)
    
    # Find the Best Overall Epoch
    best_epoch = df.loc[df['SSIM'].idxmax()]['Epoch']
    print(f"\nDone! CSV and Plot saved to {args.out_dir}")
    print(f"--> Based on SSIM, the most accurate model is from Epoch {int(best_epoch)}")

if __name__ == "__main__":
    main()