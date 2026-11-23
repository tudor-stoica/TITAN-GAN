"""
Paired Wilcoxon test, TITAN-GAN vs Vanilla-TITAN at 32^3, on the same samples and
checkpoints as exp7_conditional_fidelity.py.
"""
import sys, glob, os
import numpy as np
import torch
from scipy.stats import wilcoxon
from skimage.filters import frangi
from skimage.measure import label, regionprops

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from phase5_train_GAN import UNet3D_Generator

DEV = torch.device("cuda")
DATA = f"{ROOT}/output/phase1_data"
CKPT = {
    "TITAN":   f"{ROOT}/output/models/tighterTitan/cgan_gen_epoch_350.pth",
    "Vanilla": f"{ROOT}/output/exp2_vanilla_cgan/vanilla_gen_epoch_300.pth",
}
N = 100

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

def main():
    mask_files = sorted(glob.glob(os.path.join(DATA, "*_mask.npy")))[:N]

    gens = {}
    for k, p in CKPT.items():
        g = UNet3D_Generator().to(DEV)
        g.load_state_dict(torch.load(p, map_location=DEV))
        g.eval()
        gens[k] = g

    out = {k: {"dice": [], "b0err": [], "lcrerr": []} for k in gens}

    print(f"Conditional topology fidelity, {N} samples @ 32^3 (TITAN vs Vanilla, gap-fill)\n")
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

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{N}")

    print(f"\n{'Model':<12}{'Mask-adherence Dice':>22}{'|Betti0 err|':>15}{'|LCR err|':>12}")
    print("-" * 61)
    for k in gens:
        o = out[k]
        print(f"{k:<12}{np.mean(o['dice']):>16.4f} ±{np.std(o['dice']):.3f}"
              f"{np.mean(o['b0err']):>15.2f}{np.mean(o['lcrerr']):>12.4f}")

    print("\n--- paired Wilcoxon, TITAN vs Vanilla (n=%d) ---" % N)
    for metric, better in [("dice", "higher"), ("b0err", "lower"), ("lcrerr", "lower")]:
        t = np.array(out["TITAN"][metric]); v = np.array(out["Vanilla"][metric])
        stat, p = wilcoxon(t, v)
        win = (t > v).sum() if better == "higher" else (t < v).sum()
        print(f"  {metric:<8} TITAN better on {win}/{N} samples   p = {p:.3e}")

    np.savez(os.path.join("output/diagnostics", "conditional_fidelity_titan_vanilla_32_raw.npz"),
              **{f"{k}_{m}": np.array(out[k][m]) for k in out for m in ("dice", "b0err", "lcrerr")})
    print("\nSaved raw per-sample arrays to output/diagnostics/conditional_fidelity_titan_vanilla_32_raw.npz")

if __name__ == "__main__":
    main()
