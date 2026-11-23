"""
Held-out evaluation at 32^3.

Rebuilds the patient-level split (see make_original_split_hardneg.py) and evaluates the
train-split TITAN-GAN and Vanilla-TITAN generators on the held-out test patients:
SSIM/PSNR against the paired real CT, conditional mask-adherence Dice, per-sample
Betti-0 and LCR errors, and paired Wilcoxon tests. The epoch for each model is chosen
by SSIM on training patients only. Generators trained on all patches are evaluated on
the same patients for an in-sample comparison.

Also compares the frozen Soft-Bridge prediction with the true Frangi + CCA extraction
on real CT and on generated CT.
"""
import os, sys, glob, json, random, argparse, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from make_downstream_split import uid_of
from phase5_train_GAN import UNet3D_Generator, SoftGraphBridge
from exp7_conditional_fidelity import reverse_frangi, topo_stats, dice

DEV = torch.device("cuda")
POS_DIR = f"{ROOT}/output/phase1_data"
ORIG_NEG_DIR = f"{ROOT}/output/downstream/negatives"
SUBSET_GENS = {
    "TITAN":   (f"{ROOT}/output/downstream/train_only/models/titan", "cgan_gen_epoch_*.pth"),
    "Vanilla": (f"{ROOT}/output/downstream/train_only/vanilla", "vanilla_gen_epoch_*.pth"),
}
SUBSET_BRIDGE = f"{ROOT}/output/downstream/train_only/models/soft_graph_bridge_best.pth"
FULL_GENS = {  # generators trained on all patches (in-sample comparison)
    "TITAN":   f"{ROOT}/output/models/tighterTitan/cgan_gen_epoch_350.pth",
    "Vanilla": f"{ROOT}/output/exp2_vanilla_cgan/vanilla_gen_epoch_300.pth",
}


def original_split():
    pos = sorted(glob.glob(os.path.join(POS_DIR, "*_ct.npy")))
    neg = sorted(glob.glob(os.path.join(ORIG_NEG_DIR, "*_ct.npy")))
    uids = sorted(set(map(uid_of, pos)) | set(map(uid_of, neg)))
    rng = random.Random(42)
    shuffled = list(uids)
    rng.shuffle(shuffled)
    test_uids = set(shuffled[: int(len(shuffled) * 0.2)])
    test = [p for p in pos if uid_of(p) in test_uids]
    train = [p for p in pos if uid_of(p) not in test_uids]
    return train, test


def load_gen(path):
    g = UNet3D_Generator().to(DEV)
    g.load_state_dict(torch.load(path, map_location=DEV))
    g.eval()
    return g


def load_pair(ct_path):
    return np.load(ct_path), np.load(ct_path.replace("_ct.npy", "_mask.npy"))


@torch.no_grad()
def generate(g, mask):
    mt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(DEV)
    return g(mt).squeeze().cpu().numpy()


@torch.no_grad()
def bridge_bin(bridge, ct):
    t = torch.from_numpy(np.ascontiguousarray(ct)).unsqueeze(0).unsqueeze(0).float().to(DEV)
    return (bridge(t).squeeze().cpu().numpy() > 0.5)


def select_epoch(ckpt_dir, pattern, train_files):
    scores = {}
    for ck in sorted(glob.glob(os.path.join(ckpt_dir, pattern))):
        ep = int(ck.split("_")[-1].split(".")[0])
        g = load_gen(ck)
        s = [ssim(ct, generate(g, m), data_range=1.0) for ct, m in map(load_pair, train_files)]
        scores[ep] = float(np.mean(s))
        print(f"    epoch {ep}: train-SSIM {scores[ep]:.4f}", flush=True)
    best = max(scores, key=scores.get)
    return best, scores


def eval_gen(g, files, bridge=None):
    out = {k: [] for k in ["ssim", "psnr", "dice", "b0err", "lcrerr",
                           "bridge_dice", "frangi_dice", "bridge_frangi_agree"]}
    for ct, m in map(load_pair, files):
        fake = generate(g, m)
        mb = m > 0.5
        rec = reverse_frangi(fake)
        b0r, lr = topo_stats(mb)
        b0g, lg = topo_stats(rec)
        out["ssim"].append(ssim(ct, fake, data_range=1.0))
        out["psnr"].append(psnr(ct, fake, data_range=1.0))
        out["dice"].append(dice(rec, mb))
        out["b0err"].append(abs(b0g - b0r))
        out["lcrerr"].append(abs(lg - lr))
        if bridge is not None:
            bb = bridge_bin(bridge, fake)
            out["bridge_dice"].append(dice(bb, mb))
            out["frangi_dice"].append(dice(rec, mb))
            out["bridge_frangi_agree"].append(dice(bb, rec))
    return {k: np.array(v, dtype=float) for k, v in out.items() if v}


def eval_real_bridge(bridge, files):
    out = {"bridge_dice": [], "frangi_dice": [], "bridge_frangi_agree": []}
    for ct, m in map(load_pair, files):
        mb = m > 0.5
        bb = bridge_bin(bridge, ct)
        rec = reverse_frangi(ct)
        out["bridge_dice"].append(dice(bb, mb))
        out["frangi_dice"].append(dice(rec, mb))
        out["bridge_frangi_agree"].append(dice(bb, rec))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def ms(a):
    return f"{a.mean():.4f} ±{a.std():.3f}"


def paired(t, v, higher_better):
    wins = int((t > v).sum() if higher_better else (t < v).sum())
    ties = int((t == v).sum())
    try:
        p = float(wilcoxon(t, v).pvalue)
    except ValueError:
        p = float("nan")
    return wins, ties, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit held-out samples (0 = all; use small n to pilot)")
    ap.add_argument("--n-select", type=int, default=100, help="train positives used for epoch selection")
    ap.add_argument("--out-dir", default=f"{ROOT}/output/heldout_32")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    train, test = original_split()
    if args.n:
        test = test[: args.n]
    sel = train[: args.n_select]
    print(f"Original split rebuilt: {len(train)} train positives / {len(test)} held-out test positives "
          f"(epoch selection on {len(sel)} train positives)", flush=True)

    summary = {"n_heldout": len(test), "n_select": len(sel), "selected_epoch": {},
               "selection_ssim_by_epoch": {}, "heldout": {}, "insample_fulldata_gens": {},
               "paired_titan_vs_vanilla_heldout": {}, "soft_bridge_check": {}}
    raw = {}

    bridge = SoftGraphBridge().to(DEV)
    bridge.load_state_dict(torch.load(SUBSET_BRIDGE, map_location=DEV))
    bridge.eval()

    held = {}
    for name, (d, pat) in SUBSET_GENS.items():
        print(f"\n[{name}] selecting epoch on train positives...", flush=True)
        ep, scores = select_epoch(d, pat, sel)
        summary["selected_epoch"][name] = ep
        summary["selection_ssim_by_epoch"][name] = scores
        ck = sorted(glob.glob(os.path.join(d, pat.replace("*", f"{ep:03d}"))))[0]
        print(f"[{name}] selected epoch {ep} ({os.path.basename(ck)}); evaluating on held-out...", flush=True)
        held[name] = eval_gen(load_gen(ck), test, bridge=bridge)
        for k, v in held[name].items():
            raw[f"heldout_{name}_{k}"] = v

    print("\n[in-sample reference] full-data generators on the same patches...", flush=True)
    insample = {}
    for name, ck in FULL_GENS.items():
        insample[name] = eval_gen(load_gen(ck), test)
        for k, v in insample[name].items():
            raw[f"insample_{name}_{k}"] = v

    real_bridge = eval_real_bridge(bridge, test)
    for k, v in real_bridge.items():
        raw[f"real_{k}"] = v

    # ---- report ----
    lines = []
    lines.append(f"Held-out 32^3 evaluation: {len(test)} patients unseen by the train-split generators "
                 f"(original seed-42 split). Selected epochs (by train-SSIM): {summary['selected_epoch']}")
    lines.append("")
    hdr = f"{'Model':<26}{'SSIM':>16}{'PSNR':>16}{'Mask-adh. Dice':>18}{'|B0 err|':>16}{'|LCR err|':>16}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for tag, res in [("held-out", held), ("in-sample (full-data gen)", insample)]:
        for name, r in res.items():
            label = f"{name} [{tag}]"
            lines.append(f"{label:<26}{ms(r['ssim']):>16}{ms(r['psnr']):>16}{ms(r['dice']):>18}"
                         f"{ms(r['b0err']):>16}{ms(r['lcrerr']):>16}")
            key = "heldout" if tag == "held-out" else "insample_fulldata_gens"
            summary[key][name] = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in r.items()}
    lines.append("")
    lines.append(f"Paired Wilcoxon, TITAN vs Vanilla, held-out (n={len(test)}):")
    for metric, hb in [("ssim", True), ("psnr", True), ("dice", True), ("b0err", False), ("lcrerr", False)]:
        w, ties, p = paired(held["TITAN"][metric], held["Vanilla"][metric], hb)
        summary["paired_titan_vs_vanilla_heldout"][metric] = {"titan_wins": w, "ties": ties, "p": p}
        lines.append(f"  {metric:<8} TITAN better on {w}/{len(test)} (ties {ties})   p = {p:.3e}")
    lines.append("")
    lines.append("Soft-Bridge check, held-out patients, train-split Soft-Bridge:")
    lines.append(f"{'Input':<12}{'Bridge Dice vs mask':>22}{'Frangi Dice vs mask':>22}{'Bridge-Frangi agree':>22}")
    rows = [("Real CT", real_bridge), ("TITAN", held["TITAN"]), ("Vanilla", held["Vanilla"])]
    for label, r in rows:
        lines.append(f"{label:<12}{ms(r['bridge_dice']):>22}{ms(r['frangi_dice']):>22}{ms(r['bridge_frangi_agree']):>22}")
        summary["soft_bridge_check"][label] = {k: {"mean": float(r[k].mean()), "std": float(r[k].std())}
                                               for k in ["bridge_dice", "frangi_dice", "bridge_frangi_agree"]}
    lines.append("(Real CT's Frangi Dice is 1.0 by construction: the mask IS Frangi+CCA of that CT.)")
    lines.append(f"\nElapsed: {time.time() - t0:.1f}s")

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out_dir, "per_sample.npz"), **raw,
             heldout_files=np.array([os.path.basename(p) for p in test]))


if __name__ == "__main__":
    main()
