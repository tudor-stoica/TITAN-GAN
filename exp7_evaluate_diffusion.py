import os
import glob
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

def setup_args():
    parser = argparse.ArgumentParser(description="Exp 7: Evaluate mask-conditioned 3D DDPM checkpoints, mint final synthetic set, time inference")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data", help="Real (mask, CT) pairs to condition on / compare against")
    parser.add_argument("--models-dir", type=str, default="./output/exp7_diffusion")
    parser.add_argument("--out-dir", type=str, default="./output/exp7_diffusion")
    parser.add_argument("--num-train-timesteps", type=int, default=500)
    parser.add_argument("--num-inference-steps", type=int, default=40, help="DDIM sampling steps")
    parser.add_argument("--num-eval", type=int, default=20, help="Real pairs used per checkpoint for SSIM/PSNR sweep")
    parser.add_argument("--num-final", type=int, default=100, help="Final synthetic (ct,mask) pairs to mint from best checkpoint")
    return parser.parse_args()

def build_model(device):
    model = DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 128), num_res_blocks=1,
        attention_levels=(False, False, False, True), num_head_channels=32,
    ).to(device)
    return model

@torch.no_grad()
def ddim_sample(model, mask_tensor, ddim_scheduler, device, num_inference_steps):
    """One DDIM sampling loop, mask-conditioned via channel concat. mask_tensor is expected
    in raw [0,1]; rescaled to [-1,1] here to match training. Returns CT in [0,1]."""
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
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(glob.glob(os.path.join(args.models_dir, "diffusion_unet_epoch_*.pth")))
    if not checkpoints:
        print("Error: No diffusion checkpoints found.")
        return

    real_mask_files = sorted(glob.glob(os.path.join(args.data_dir, "*_mask.npy")))[:args.num_eval]
    real_ct_files = [f.replace("_mask.npy", "_ct.npy") for f in real_mask_files]

    ddim_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_timesteps, schedule="linear_beta",
                                    clip_sample=True, prediction_type="epsilon")

    print(f"Evaluating {len(checkpoints)} diffusion checkpoints, {args.num_inference_steps} DDIM steps, {len(real_mask_files)} eval samples each...")
    results = []
    model = build_model(device)

    for ckpt_path in checkpoints:
        epoch_num = int(os.path.basename(ckpt_path).split("_")[-1].replace(".pth", ""))
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        ssim_scores, psnr_scores = [], []
        for mf, cf in tqdm(zip(real_mask_files, real_ct_files), total=len(real_mask_files), desc=f"Epoch {epoch_num}", leave=False):
            mask_np = np.load(mf)
            real_ct_np = np.load(cf)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)

            fake_ct = ddim_sample(model, mask_t, ddim_scheduler, device, args.num_inference_steps)
            fake_ct_np = fake_ct.squeeze().cpu().numpy()

            ssim_scores.append(ssim(real_ct_np, fake_ct_np, data_range=1.0))
            psnr_scores.append(psnr(real_ct_np, fake_ct_np, data_range=1.0))

        avg_ssim, avg_psnr = float(np.mean(ssim_scores)), float(np.mean(psnr_scores))
        results.append({"Epoch": epoch_num, "SSIM": avg_ssim, "PSNR": avg_psnr})
        print(f"Epoch {epoch_num} -> SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f}dB")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "diffusion_evaluation.csv"), index=False)
    best_epoch = int(df.loc[df["SSIM"].idxmax()]["Epoch"])
    print(f"\nEvaluation complete! Best diffusion epoch: {best_epoch}")

    # ---- Timing: per-sample DDIM inference wall-clock ----
    best_ckpt_path = os.path.join(args.models_dir, f"diffusion_unet_epoch_{best_epoch:03d}.pth")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    print(f"\nTiming DDIM inference ({args.num_inference_steps} steps) over {min(20, len(real_mask_files))} warmed-up samples...")
    timing_mask_files = real_mask_files[:20] if len(real_mask_files) >= 20 else real_mask_files
    # warmup (exclude first-call CUDA kernel compilation overhead from the measurement)
    warm_mask = torch.from_numpy(np.load(timing_mask_files[0])).unsqueeze(0).unsqueeze(0).float().to(device)
    _ = ddim_sample(model, warm_mask, ddim_scheduler, device, args.num_inference_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()

    per_sample_times = []
    for mf in timing_mask_files:
        mask_t = torch.from_numpy(np.load(mf)).unsqueeze(0).unsqueeze(0).float().to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = ddim_sample(model, mask_t, ddim_scheduler, device, args.num_inference_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_sample_times.append(time.time() - t0)

    avg_inference_time = float(np.mean(per_sample_times))
    std_inference_time = float(np.std(per_sample_times))
    print(f"Per-sample DDIM inference time: {avg_inference_time:.3f}s +/- {std_inference_time:.3f}s (n={len(per_sample_times)})")

    with open(os.path.join(args.out_dir, "inference_timing.txt"), "w") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"num_inference_steps={args.num_inference_steps}\n")
        f.write(f"avg_inference_seconds={avg_inference_time:.4f}\n")
        f.write(f"std_inference_seconds={std_inference_time:.4f}\n")
        f.write(f"n_timed_samples={len(per_sample_times)}\n")

    # ---- Mint final synthetic (ct, mask) pairs for topology / mask-adherence evaluation ----
    synth_dir = os.path.join(args.out_dir, "synthetic_data")
    if os.path.exists(synth_dir):
        shutil.rmtree(synth_dir)
    os.makedirs(synth_dir)

    print(f"\nMinting {args.num_final} synthetic (ct, mask) pairs from best checkpoint (epoch {best_epoch})...")
    all_mask_files = sorted(glob.glob(os.path.join(args.data_dir, "*_mask.npy")))
    rng = np.random.RandomState(123)
    mint_indices = rng.choice(len(all_mask_files), size=min(args.num_final, len(all_mask_files)), replace=(args.num_final > len(all_mask_files)))

    for i, idx in enumerate(tqdm(mint_indices, desc="Minting diffusion CTs")):
        mf = all_mask_files[idx]
        mask_np = np.load(mf)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
        fake_ct = ddim_sample(model, mask_t, ddim_scheduler, device, args.num_inference_steps)
        fake_ct_np = fake_ct.squeeze().cpu().numpy().astype(np.float32)

        np.save(os.path.join(synth_dir, f"diffusion_{i:04d}_ct.npy"), fake_ct_np)
        np.save(os.path.join(synth_dir, f"diffusion_{i:04d}_mask.npy"), mask_np.astype(np.float32))

    print(f"\nSuccess! Diffusion synthetic data secured at {synth_dir}")

if __name__ == "__main__":
    main()
