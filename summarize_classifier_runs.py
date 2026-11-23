"""
Summarize classifier runs: mean and std of each metric over seeds.
"""
import os, re, json, glob, sys, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["fulldata", "lowdata"])
ap.add_argument("--runs-dir", required=True)
args = ap.parse_args()
METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "AUC"]

rows = {}
for f in sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.json"))):
    name = os.path.basename(os.path.dirname(f))
    m = json.load(open(f))
    if args.mode == "fulldata":
        mm = re.match(r"^(.*)_s(\d+)$", name)
        if not mm: continue
        key = mm.group(1)
    else:
        mm = re.match(r"^(real_only|vanilla|titan|alpha_wgan)_n(\d+)_s(\d+)$", name)
        if not mm: continue
        key = (int(mm.group(2)), mm.group(1))
    rows.setdefault(key, []).append(m)

order = ["real_only", "vanilla", "titan", "alpha_wgan", "diffusion"]
def fmt(ms, k):
    v = np.array([m[k] for m in ms])
    return f"{v.mean():.3f}±{v.std():.3f}"

print(f"{'condition':<22}{'n':>3}  " + "  ".join(f"{k:>13}" for k in METRICS) + "   n_real/n_synth/N_pos/N_neg")
for key in sorted(rows, key=lambda k: (k[0], order.index(k[1])) if isinstance(k, tuple) else order.index(k) if k in order else 99):
    ms = rows[key]
    label = key if isinstance(key, str) else f"N={key[0]} {key[1]}"
    info = sorted({(m["n_real_train_pos"], m["n_synth_augment"], m["N_pos"], m["N_neg"]) for m in ms})
    print(f"{label:<22}{len(ms):>3}  " + "  ".join(f"{fmt(ms, k):>13}" for k in METRICS) + f"   {info}")
