import os
import glob
import shutil
import argparse
import numpy as np

def setup_args():
    parser = argparse.ArgumentParser(description="Subsample N real positive patches from the existing train-positive pool (no test-set contact).")
    parser.add_argument("--source-dir", type=str, default="./output/downstream/split/real_train_pos")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(args.source_dir, "*_mask.npy")))
    assert len(mask_files) >= args.n, f"Only {len(mask_files)} available, need {args.n}"

    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(len(mask_files), size=args.n, replace=False)

    for i in chosen:
        mf = mask_files[i]
        cf = mf.replace("_mask.npy", "_ct.npy")
        shutil.copy(mf, os.path.join(args.out_dir, os.path.basename(mf)))
        shutil.copy(cf, os.path.join(args.out_dir, os.path.basename(cf)))

    print(f"Copied {args.n} positive (ct,mask) pairs to {args.out_dir}")

if __name__ == "__main__":
    main()
