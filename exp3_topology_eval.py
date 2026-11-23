"""
Topology evaluation: runs Frangi + CCA on real and generated CT patches and reports
mean Betti-0 (number of components) and LCR (largest component ratio) per dataset.
Patches are processed in parallel with multiprocessing.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from skimage.filters import frangi
from skimage.measure import label, regionprops

def apply_frangi_and_sanitize(ct_patch, frangi_thresh=0.005, cca_thresh=75):
    vesselness = frangi(ct_patch, sigmas=range(1, 3, 1), black_ridges=False)
    binary_mask = vesselness > frangi_thresh
    labeled_mask = label(binary_mask, connectivity=3)
    clean_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for region in regionprops(labeled_mask):
        if region.area >= cca_thresh:
            coords = region.coords
            clean_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return clean_mask

def calculate_topology_metrics(binary_mask):
    labeled_mask = label(binary_mask, connectivity=3)
    regions = regionprops(labeled_mask)
    betti_0 = len(regions)
    if betti_0 == 0:
        return 0, 0.0
    total_volume = sum([r.area for r in regions])
    largest_volume = max([r.area for r in regions])
    lcr = largest_volume / total_volume
    return betti_0, lcr

def _worker(args):
    file_path, frangi_thresh, cca_thresh = args
    ct_np = np.load(file_path)
    if ct_np.min() < 0:
        ct_np = np.clip(ct_np, -1000, 400)
        ct_np = (ct_np + 1000) / 1400.0
    clean_mask = apply_frangi_and_sanitize(ct_np, frangi_thresh, cca_thresh)
    return calculate_topology_metrics(clean_mask)

def process_dataset(name, file_list, args, pool):
    if not file_list:
        print(f"Warning: No files found for {name}")
        return None
    print(f"Processing {name} (N={len(file_list)})...")
    tasks = [(f, args.frangi_thresh, args.cca_thresh) for f in file_list]
    results = pool.map(_worker, tasks)
    betti_list = [r[0] for r in results]
    lcr_list = [r[1] for r in results]
    return {"Dataset": name, "Mean_Betti_0": np.mean(betti_list), "Mean_LCR": np.mean(lcr_list)}

def setup_args():
    parser = argparse.ArgumentParser(description="Exp 3 (parallel): Reverse Frangi Topology Test")
    parser.add_argument("--real-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--baseline-dir", type=str, default="./output/exp1_baseline/synthetic_data")
    parser.add_argument("--vanilla-dir", type=str, default="./output/exp2_vanilla_cgan/synthetic_data")
    parser.add_argument("--cascading-dir", type=str, default="./output/phase7_final_dataset_NEW")
    parser.add_argument("--alpha-wgan-dir", type=str, default=None)
    parser.add_argument("--diffusion-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="./output/exp3_topology_resultsNEW")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--frangi-thresh", type=float, default=0.005)
    parser.add_argument("--cca-thresh", type=int, default=75)
    parser.add_argument("--cores", type=int, default=max(1, cpu_count() - 2))
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)

    real_files = sorted(glob.glob(os.path.join(args.real_dir, "*_ct.npy")))[:args.num_samples]
    baseline_files = sorted(glob.glob(os.path.join(args.baseline_dir, "*.npy")))[:args.num_samples]
    vanilla_files = sorted(glob.glob(os.path.join(args.vanilla_dir, "*.npy")))[:args.num_samples]
    cascading_files = sorted(glob.glob(os.path.join(args.cascading_dir, "*_ct.npy")))[:args.num_samples]

    dataset_list = [("Real Phase 1", real_files),
                     ("Baseline WGAN-GP", baseline_files),
                     ("Vanilla cGAN", vanilla_files),
                     ("Cascading cGAN", cascading_files)]

    if args.alpha_wgan_dir:
        alpha_wgan_files = sorted(glob.glob(os.path.join(args.alpha_wgan_dir, "*.npy")))[:args.num_samples]
        dataset_list.append(("Alpha-WGAN", alpha_wgan_files))

    if args.diffusion_dir:
        diffusion_files = sorted(glob.glob(os.path.join(args.diffusion_dir, "*_ct.npy")))[:args.num_samples]
        dataset_list.append(("Diffusion (DDPM)", diffusion_files))

    print(f"Using {args.cores} worker processes")
    results = []
    with Pool(args.cores) as pool:
        for name, files in dataset_list:
            res = process_dataset(name, files, args, pool)
            if res:
                results.append(res)
                print(f"  {name}: Betti-0={res['Mean_Betti_0']:.2f}, LCR={res['Mean_LCR']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "topology_comparison.csv"), index=False)
    print("\n--- Final Topology Results ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
