"""
Hard negative mining.

Scores a pool of LUNA16 candidates (candidates_V2.csv) by the mean Frangi vesselness
response at the candidate center and keeps the highest-scoring ones as negatives. These
vessel-crossing candidates are a common source of false positives and are harder to
separate from nodules than randomly drawn candidates.

Two passes: pass 1 extracts and scores the pool (patches discarded, to keep
multiprocessing payloads small); pass 2 re-extracts and saves the --num-samples
hardest candidates.
"""
import os
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi
import multiprocessing

def setup_args():
    parser = argparse.ArgumentParser(description="Phase 8 (hard): Extract vessel-crossing-biased Negative LUNA16 Candidates")
    parser.add_argument("--luna-dir", type=str, default="./data/luna16", help="Path to LUNA16 dataset root")
    parser.add_argument("--candidates-csv", type=str, default="./data/luna16/candidates_V2/candidates_V2.csv",
                         help="Candidates file to sample negatives from (candidates_V2.csv, not the original candidates.csv)")
    parser.add_argument("--output-dir", type=str, default="./output/downstream/negatives_hard")
    parser.add_argument("--pool-size", type=int, default=4000, help="Candidates to extract+score before selecting the hardest")
    parser.add_argument("--num-samples", type=int, default=1200, help="Number of hardest (highest Frangi response) negatives to keep")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cores", type=int, default=10)
    return parser.parse_args()

def extract_resampled_crop(image, center_physical, output_size=(32, 32, 32), output_spacing=(1.0, 1.0, 1.0)):
    """Same as extract_resampled_crop in phase1_generate_masks.py."""
    origin = [c - (s * sp) / 2.0 for c, s, sp in zip(center_physical, output_size, output_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(-1000)
    resample.SetInterpolator(sitk.sitkBSpline)

    cropped_sitk = resample.Execute(image)
    return sitk.GetArrayFromImage(cropped_sitk)

def normalize_ct(ct_patch):
    """Same normalization as phase1_generate_masks.py's apply_frangi_and_sanitize (minus the Frangi/mask step)."""
    clipped_ct = np.clip(ct_patch, -1000, 400)
    return (clipped_ct + 1000) / 1400.0

def frangi_score(normalized_ct):
    """Mean Frangi vesselness response: higher = more tubular/vessel-crossing-like
    (i.e. the classic source of nodule-CAD false positives), so harder to tell
    apart from a real nodule cross-section."""
    v = frangi(normalized_ct, sigmas=range(1, 3, 1), black_ridges=False)
    return float(np.mean(v))

def score_worker(task_args):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    uid, center_physical, mhd_path, patch_size, pool_idx = task_args
    try:
        itk_image = sitk.ReadImage(mhd_path)
        ct_patch = extract_resampled_crop(itk_image, center_physical, output_size=(patch_size, patch_size, patch_size))
        normalized_ct = normalize_ct(ct_patch)
        score = frangi_score(normalized_ct)
        return True, pool_idx, uid, score
    except Exception as e:
        return False, pool_idx, f"Failed on {uid}: {str(e)}", None

def save_worker(task_args):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    uid, center_physical, mhd_path, output_dir, patch_size, processed_idx = task_args
    try:
        itk_image = sitk.ReadImage(mhd_path)
        ct_patch = extract_resampled_crop(itk_image, center_physical, output_size=(patch_size, patch_size, patch_size))
        normalized_ct = normalize_ct(ct_patch)
        save_path = os.path.join(output_dir, f"negative_{processed_idx:04d}_{uid}_ct.npy")
        np.save(save_path, normalized_ct.astype(np.float32))
        return True, uid
    except Exception as e:
        return False, f"Failed on {uid}: {str(e)}"

def main():
    args = setup_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    print(f"Loading {args.candidates_csv}...")
    df = pd.read_csv(args.candidates_csv)
    df = df[df['class'] == 0]
    print(f"Found {len(df)} class=0 candidates total.")

    mhd_files = glob.glob(os.path.join(args.luna_dir, "subset*", "**", "*.mhd"), recursive=True)
    uid_to_path = {os.path.basename(f).replace(".mhd", ""): f for f in mhd_files}
    print(f"Found {len(uid_to_path)} locally available scans.")

    df = df[df['seriesuid'].isin(uid_to_path.keys())]
    print(f"{len(df)} class=0 candidates map to locally available scans.")

    pool_n = min(args.pool_size, len(df))
    pool_df = df.sample(n=pool_n, random_state=args.seed)
    print(f"Scoring a pool of {pool_n} candidates by Frangi vesselness response "
          f"(will keep the hardest {args.num_samples})...")

    score_tasks = []
    for pool_idx, (_, row) in enumerate(pool_df.iterrows()):
        uid = row['seriesuid']
        center_physical = (row['coordX'], row['coordY'], row['coordZ'])
        mhd_path = uid_to_path[uid]
        score_tasks.append((uid, center_physical, mhd_path, args.patch_size, pool_idx))

    scored = []
    with multiprocessing.Pool(processes=args.cores) as pool_exec:
        for i, (success, pool_idx, uid_or_err, score) in enumerate(pool_exec.imap_unordered(score_worker, score_tasks)):
            if success:
                scored.append((pool_idx, uid_or_err, score))
                print(f"Scoring progress: [{i + 1}/{len(score_tasks)}]", end='\r')
            else:
                print(f"\nError: {uid_or_err}")

    print(f"\nScored {len(scored)}/{len(score_tasks)} candidates successfully.")
    all_scores = np.array([s[2] for s in scored])
    print(f"Pool Frangi score stats: mean={all_scores.mean():.5f}  std={all_scores.std():.5f}  "
          f"min={all_scores.min():.5f}  max={all_scores.max():.5f}")

    scored.sort(key=lambda r: r[2], reverse=True)
    keep_n = min(args.num_samples, len(scored))
    hardest = scored[:keep_n]
    hard_scores = np.array([h[2] for h in hardest])
    print(f"Kept {keep_n} hardest negatives. Kept-set Frangi score stats: "
          f"mean={hard_scores.mean():.5f}  std={hard_scores.std():.5f}  min={hard_scores.min():.5f}")
    print(f"(Kept-set mean is {hard_scores.mean() / all_scores.mean():.2f}x the full-pool mean - "
          f"higher confirms these are genuinely more vessel-crossing-like than a random draw.)")

    save_tasks = []
    for processed_idx, (pool_idx, uid, score) in enumerate(hardest):
        row = pool_df.iloc[pool_idx]
        center_physical = (row['coordX'], row['coordY'], row['coordZ'])
        mhd_path = uid_to_path[uid]
        save_tasks.append((uid, center_physical, mhd_path, args.output_dir, args.patch_size, processed_idx))

    print(f"\nRe-extracting and saving {len(save_tasks)} hardest negative patches...")
    successful_count = 0
    with multiprocessing.Pool(processes=args.cores) as pool_exec:
        for idx, (success, msg) in enumerate(pool_exec.imap_unordered(save_worker, save_tasks)):
            if success:
                successful_count += 1
                print(f"Save progress: [{idx + 1}/{len(save_tasks)}]", end='\r')
            else:
                print(f"\nError: {msg}")

    print(f"\nSuccess! Saved {successful_count}/{len(save_tasks)} hard negative patches to {args.output_dir}")

if __name__ == "__main__":
    main()
