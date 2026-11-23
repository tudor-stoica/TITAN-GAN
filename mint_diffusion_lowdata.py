"""
Sample synthetic CT from a trained mask-conditioned DDPM, conditioned on a directory of
VAE-generated masks. Writes paired {base}_mask.npy / {base}_ct.npy files in the same format
as phase7_gen_synth_scans.py, so the output can be passed to train_classifier.py
--synth-pos-dir.
"""
import os
import glob
import argparse
import shutil
import numpy as np
import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler


@torch.no_grad()
def ddim_sample(model, mask_tensor, ddim_scheduler, device, num_inference_steps):
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
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion-ckpt", type=str, required=True)
    p.add_argument("--mask-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-inference-steps", type=int, default=40)
    p.add_argument("--num-train-timesteps", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    model = DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 128), num_res_blocks=1,
        attention_levels=(False, False, False, True), num_head_channels=32,
    ).to(device)
    model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
    model.eval()

    scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps, schedule="linear_beta",
                               clip_sample=True, prediction_type="epsilon")

    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*_mask.npy")))
    print(f"Discovered {len(mask_files)} synthetic blueprints in {args.mask_dir}")

    for mask_path in mask_files:
        mask_np = np.load(mask_path)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
        fake_ct = ddim_sample(model, mask_tensor, scheduler, device, args.num_inference_steps)
        fake_ct_np = fake_ct.squeeze().cpu().numpy()

        base_name = os.path.basename(mask_path).replace("_mask.npy", "")
        shutil.copy(mask_path, os.path.join(args.out_dir, f"{base_name}_mask.npy"))
        np.save(os.path.join(args.out_dir, f"{base_name}_ct.npy"), fake_ct_np)

    print(f"Rendered {len(mask_files)} synthetic (mask, CT) pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
