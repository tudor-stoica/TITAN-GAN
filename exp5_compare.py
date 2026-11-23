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

# Generator architecture (needed to load checkpoints)
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

# Evaluation Logic
def setup_args():
    parser = argparse.ArgumentParser(description="Exp 5: Evaluate alpha-WGAN Checkpoints")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data", help="Real CTs")
    parser.add_argument("--models-dir", type=str, default="./output/models/alpha_wgan", help="alpha-WGAN checkpoints")
    parser.add_argument("--out-dir", type=str, default="./output/exp5_alpha_wgan")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-eval", type=int, default=100, help="Samples to test per checkpoint")
    parser.add_argument("--num-final", type=int, default=600, help="Final patches to mint from Best Epoch")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "synthetic_data"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(glob.glob(os.path.join(args.models_dir, "alpha_wgan_gen_epoch_*.pth")))
    if not checkpoints:
        print("Error: No alpha-WGAN checkpoints found.")
        return

    # Load Real Data for baseline distribution comparison (same sequential-pairing
    # convention as exp1_compare.py, since generation here is unconditional)
    real_ct_files = sorted(glob.glob(os.path.join(args.data_dir, "*_ct.npy")))[:args.num_eval]
    real_cts = [np.load(f) for f in real_ct_files]

    generator = Generator3D(latent_dim=args.latent_dim, output_size=args.patch_size).to(device)
    results = []

    print(f"Evaluating {len(checkpoints)} alpha-WGAN Checkpoints...")

    for ckpt_path in checkpoints:
        epoch_num = int(ckpt_path.split('_')[-1].split('.')[0])
        generator.load_state_dict(torch.load(ckpt_path, map_location=device))
        generator.eval()

        epoch_ssim, epoch_psnr = 0.0, 0.0

        with torch.no_grad():
            for i in tqdm(range(args.num_eval), desc=f"Epoch {epoch_num}", leave=False):
                z = torch.randn(1, args.latent_dim).to(device)
                fake_ct_np = generator(z).squeeze().cpu().numpy()
                real_ct_np = real_cts[i]

                epoch_ssim += ssim(real_ct_np, fake_ct_np, data_range=1.0)
                epoch_psnr += psnr(real_ct_np, fake_ct_np, data_range=1.0)

        avg_ssim = epoch_ssim / args.num_eval
        avg_psnr = epoch_psnr / args.num_eval
        results.append({'Epoch': epoch_num, 'SSIM': avg_ssim, 'PSNR': avg_psnr})
        print(f"Epoch {epoch_num} -> Average SSIM: {avg_ssim:.4f} | Average PSNR: {avg_psnr:.2f}dB")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "alpha_wgan_evaluation.csv"), index=False)

    best_epoch = int(df.loc[df['SSIM'].idxmax()]['Epoch'])
    print(f"\nEvaluation Complete! The 'Best' alpha-WGAN model is Epoch {best_epoch}.")

    print(f"\nMinting {args.num_final} patches from Epoch {best_epoch} for topology testing...")
    best_ckpt_path = os.path.join(args.models_dir, f"alpha_wgan_gen_epoch_{best_epoch:03d}.pth")
    generator.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    generator.eval()

    with torch.no_grad():
        for i in tqdm(range(args.num_final), desc="Generating alpha-WGAN CTs"):
            z = torch.randn(1, args.latent_dim).to(device)
            fake_ct = generator(z).squeeze().cpu().numpy()
            out_path = os.path.join(args.out_dir, "synthetic_data", f"alpha_wgan_ct_{i:04d}.npy")
            np.save(out_path, fake_ct)

    print(f"\nSuccess! alpha-WGAN data secured at {os.path.join(args.out_dir, 'synthetic_data')}")

if __name__ == "__main__":
    main()
