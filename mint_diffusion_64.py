"""
Sample synthetic (ct, mask) pairs from the 64^3 diffusion checkpoint, conditioned on the
same VAE-generated masks (output/res64/phase4_synthetic_data) used for the other 64^3 models.
"""
import os
import glob
import argparse
import numpy as np
import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler

def build_model(device):
    return DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 128), num_res_blocks=1,
        attention_levels=(False, False, False, True), num_head_channels=32,
    ).to(device)

@torch.no_grad()
def ddim_sample(model, mask_tensor, ddim_scheduler, device, num_inference_steps=40):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, default="./output/res64/phase4_synthetic_data")
    parser.add_argument("--ckpt", type=str, default="./output/diffusion_64/diffusion_unet_epoch_1300.pth")
    parser.add_argument("--out-dir", type=str, default="./output/diffusion_64/synthetic_data")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    ddim_scheduler = DDIMScheduler(num_train_timesteps=500, schedule="linear_beta", clip_sample=True)

    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.npy")))
    print(f"Minting {len(mask_files)} diffusion CTs @ 64^3 from {args.mask_dir}")

    for i, mf in enumerate(mask_files):
        mask = np.load(mf)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
        ct = ddim_sample(model, mask_t, ddim_scheduler, device, args.num_inference_steps)
        ct_np = ct.squeeze().cpu().numpy()

        tag = f"diff64_{i:04d}"
        np.save(os.path.join(args.out_dir, f"{tag}_ct.npy"), ct_np)
        np.save(os.path.join(args.out_dir, f"{tag}_mask.npy"), mask)

        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(mask_files)}")

    print(f"Done. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
