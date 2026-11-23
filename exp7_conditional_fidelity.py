"""
Conditional mask-adherence at 32^3: TITAN-GAN, Vanilla-TITAN and Diffusion.
Each generated CT is passed back through Frangi + CCA and compared (Dice, Betti-0 and
LCR error) with its own conditioning mask; paired Wilcoxon test TITAN-GAN vs Diffusion.
GAN checkpoint paths below are the best epochs (by SSIM) from model selection.
"""
import sys, glob, os
import numpy as np
import torch
from scipy.stats import wilcoxon
from skimage.filters import frangi
from skimage.measure import label, regionprops
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from phase5_train_GAN import UNet3D_Generator

DEV = torch.device("cuda")
DATA = f"{ROOT}/output/phase1_data"
CKPT = {
    "TITAN":   f"{ROOT}/output/models/tighterTitan/cgan_gen_epoch_350.pth",
    "Vanilla": f"{ROOT}/output/exp2_vanilla_cgan/vanilla_gen_epoch_300.pth",
}
DIFFUSION_CKPT = None  # set from --diffusion-ckpt
N = 100
NUM_INFERENCE_STEPS = 40
NUM_TRAIN_TIMESTEPS = 500

def reverse_frangi(ct, thresh=0.005, cca=75):
    v = frangi(ct, sigmas=range(1, 3, 1), black_ridges=False)
    lm = label(v > thresh, connectivity=3)
    clean = np.zeros_like(lm, dtype=np.uint8)
    for r in regionprops(lm):
        if r.area >= cca:
            c = r.coords
            clean[c[:, 0], c[:, 1], c[:, 2]] = 1
    return clean

def topo_stats(binary):
    lm = label(binary, connectivity=3)
    areas = [r.area for r in regionprops(lm)]
    if not areas:
        return 0, 0.0
    return len(areas), max(areas) / sum(areas)

def dice(a, b, eps=1e-5):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return (2 * (a * b).sum() + eps) / (a.sum() + b.sum() + eps)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion-ckpt", type=str, required=True)
    parser.add_argument("--n", type=int, default=N)
    args = parser.parse_args()

    mask_files = sorted(glob.glob(os.path.join(DATA, "*_mask.npy")))[:args.n]

    gens = {}
    for k, p in CKPT.items():
        g = UNet3D_Generator().to(DEV)
        g.load_state_dict(torch.load(p, map_location=DEV))
        g.eval()
        gens[k] = g

    diffusion_model = DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=(32, 64, 128, 128), num_res_blocks=1,
        attention_levels=(False, False, False, True), num_head_channels=32,
    ).to(DEV)
    diffusion_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=DEV))
    diffusion_model.eval()
    ddim_scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, schedule="linear_beta",
                                    clip_sample=True, prediction_type="epsilon")

    model_names = list(gens.keys()) + ["Diffusion"]
    out = {k: {"dice": [], "b0err": [], "lcrerr": []} for k in model_names}

    print(f"Conditional topology fidelity, {args.n} samples @ 32^3 (TITAN, Vanilla, Diffusion)\n")
    for i, mf in enumerate(mask_files):
        m = np.load(mf)
        b0_req, lcr_req = topo_stats(m > 0.5)
        mt = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(DEV)

        with torch.no_grad():
            for k, g in gens.items():
                fnp = g(mt).squeeze().cpu().numpy()
                rec = reverse_frangi(fnp)
                b0_got, lcr_got = topo_stats(rec)
                out[k]["dice"].append(dice(rec, m > 0.5))
                out[k]["b0err"].append(abs(b0_got - b0_req))
                out[k]["lcrerr"].append(abs(lcr_got - lcr_req))

            fake_ct = ddim_sample(diffusion_model, mt, ddim_scheduler, DEV, NUM_INFERENCE_STEPS)
            fnp = fake_ct.squeeze().cpu().numpy()
            rec = reverse_frangi(fnp)
            b0_got, lcr_got = topo_stats(rec)
            out["Diffusion"]["dice"].append(dice(rec, m > 0.5))
            out["Diffusion"]["b0err"].append(abs(b0_got - b0_req))
            out["Diffusion"]["lcrerr"].append(abs(lcr_got - lcr_req))

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{args.n}")

    print(f"\n{'Model':<12}{'Mask-adherence Dice':>22}{'|Betti0 err|':>15}{'|LCR err|':>12}")
    print("-" * 61)
    for k in model_names:
        o = out[k]
        print(f"{k:<12}{np.mean(o['dice']):>16.4f} ±{np.std(o['dice']):.3f}"
              f"{np.mean(o['b0err']):>15.2f}{np.mean(o['lcrerr']):>12.4f}")

    print("\n--- paired Wilcoxon, TITAN vs Diffusion (n=%d) ---" % args.n)
    for metric, better in [("dice", "higher"), ("b0err", "lower"), ("lcrerr", "lower")]:
        t = np.array(out["TITAN"][metric]); d = np.array(out["Diffusion"][metric])
        stat, p = wilcoxon(t, d)
        win = (t > d).sum() if better == "higher" else (t < d).sum()
        print(f"  {metric:<8} TITAN better on {win}/{args.n} samples   p = {p:.3e}")

if __name__ == "__main__":
    main()
