import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup_args():
    parser = argparse.ArgumentParser(description="Aggregate low-data sweep classifier runs across N and seeds into a curve")
    parser.add_argument("--runs-dir", type=str, default="./output/lowdata/classifier_runs")
    parser.add_argument("--out-dir", type=str, default="./output/lowdata")
    return parser.parse_args()

CONDITION_LABELS = {
    "real_only": "Real-only",
    "vanilla": "+ Vanilla cGAN",
    "titan": "+ TITAN-GAN",
    "alpha_wgan": "+ alpha-WGAN",
}
CONDITION_COLORS = {
    "real_only": "#7f7f7f",
    "vanilla": "#4c72b0",
    "titan": "#c44e52",
    "alpha_wgan": "#55a868",
}

# matches e.g. "titan_n25_s43" or "real_only_n150_s42"
PATTERN = re.compile(r"^(real_only|vanilla|titan|alpha_wgan)_n(\d+)_s(\d+)$")

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        name = os.path.basename(run_dir)
        m = PATTERN.match(name)
        if not m:
            continue
        condition, n_str, seed_str = m.groups()
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Warning: no metrics.json in {run_dir}, skipping")
            continue
        with open(metrics_path) as f:
            d = json.load(f)
        rows.append({
            "condition": condition,
            "N": int(n_str),
            "seed": int(seed_str),
            "Accuracy": d["Accuracy"],
            "Sensitivity": d["Sensitivity"],
            "Specificity": d["Specificity"],
            "Precision": d["Precision"],
            "AUC": d["AUC"],
        })

    if not rows:
        print("No matching classifier runs found.")
        return

    df = pd.DataFrame(rows)
    # N=100 has only 2 seeds and is not plotted.
    df = df[df["N"] != 100]
    df.to_csv(os.path.join(args.out_dir, "lowdata_curve_raw.csv"), index=False)
    print(f"Loaded {len(df)} runs across N={sorted(df['N'].unique())}, conditions={sorted(df['condition'].unique())}")

    agg = df.groupby(["condition", "N"]).agg(
        Specificity_mean=("Specificity", "mean"),
        Specificity_std=("Specificity", "std"),
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    agg["Specificity_std"] = agg["Specificity_std"].fillna(0.0)
    agg["AUC_std"] = agg["AUC_std"].fillna(0.0)
    agg.to_csv(os.path.join(args.out_dir, "lowdata_curve_summary.csv"), index=False)
    print("\n--- Summary (mean +/- std across seeds) ---")
    print(agg.to_string(index=False))

    # ---- Plot: Specificity vs N, one line per condition, shaded std band ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for condition in ["real_only", "vanilla", "titan", "alpha_wgan"]:
        sub = agg[agg["condition"] == condition].sort_values("N")
        if sub.empty:
            continue
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]
        # mark points with only 1 seed distinctly (no meaningful std band)
        single_seed = sub["n_seeds"] <= 1

        ax1.plot(sub["N"], sub["Specificity_mean"], marker="o", color=color, label=label, linewidth=2)
        if single_seed.any():
            ax1.scatter(sub.loc[single_seed, "N"], sub.loc[single_seed, "Specificity_mean"],
                        marker="x", color=color, s=80, zorder=5)

        ax2.plot(sub["N"], sub["AUC_mean"], marker="o", color=color, label=label, linewidth=2)
        if single_seed.any():
            ax2.scatter(sub.loc[single_seed, "N"], sub.loc[single_seed, "AUC_mean"],
                        marker="x", color=color, s=80, zorder=5)

    ax1.set_xlabel("N real positive training patches", fontweight="bold")
    ax1.set_ylabel("Specificity (true-negative rate)", fontweight="bold")
    ax1.set_title("Specificity vs. Real-Data Scarcity\n(x = single-seed point)", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("N real positive training patches", fontweight="bold")
    ax2.set_ylabel("AUC", fontweight="bold")
    ax2.set_title("AUC vs. Real-Data Scarcity", fontweight="bold")
    ax2.set_ylim(0.8, 1.0)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Low-Data Synthetic Augmentation Sweep", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "lowdata_curve.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()
