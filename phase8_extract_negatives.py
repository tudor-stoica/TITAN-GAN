import os
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
import SimpleITK as sitk
import multiprocessing

def setup_args():
    parser = argparse.ArgumentParser(description="Phase 8: Extract Negative (non-nodule) LUNA16 Candidates")
    parser.add_argument("--luna-dir", type=str, default="./data/luna16", help="Path to LUNA16 dataset root")
    parser.add_argument("--candidates-csv", type=str, default="./data/luna16/candidates_V2/candidates_V2.csv",
                         help="Candidates file to sample negatives from (candidates_V2.csv, not the original candidates.csv)")
    parser.add_argument("--output-dir", type=str, default="./output/downstream/negatives")
    parser.add_argument("--num-samples", type=int, default=1200, help="Number of class=0 candidates to extract")
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

def worker_process(task_args):
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

    # Discover which scans are actually available locally (subset0-4, etc.)
    mhd_files = glob.glob(os.path.join(args.luna_dir, "subset*", "**", "*.mhd"), recursive=True)
    uid_to_path = {os.path.basename(f).replace(".mhd", ""): f for f in mhd_files}
    print(f"Found {len(uid_to_path)} locally available scans.")

    df = df[df['seriesuid'].isin(uid_to_path.keys())]
    print(f"{len(df)} class=0 candidates map to locally available scans.")

    if len(df) < args.num_samples:
        print(f"Warning: only {len(df)} candidates available, requested {args.num_samples}. Using all available.")
        sampled = df
    else:
        sampled = df.sample(n=args.num_samples, random_state=args.seed)

    tasks = []
    for idx, (_, row) in enumerate(sampled.iterrows()):
        uid = row['seriesuid']
        center_physical = (row['coordX'], row['coordY'], row['coordZ'])
        mhd_path = uid_to_path[uid]
        tasks.append((uid, center_physical, mhd_path, args.output_dir, args.patch_size, idx))

    print(f"Extracting {len(tasks)} negative patches across {args.cores} core(s)...")

    successful_count = 0
    with multiprocessing.Pool(processes=args.cores) as pool:
        for idx, (success, msg) in enumerate(pool.imap_unordered(worker_process, tasks)):
            if success:
                successful_count += 1
                print(f"Progress: [{idx + 1}/{len(tasks)}] completed.", end='\r')
            else:
                print(f"\nError: {msg}")

    print(f"\nSuccess! Extracted {successful_count}/{len(tasks)} negative patches. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
