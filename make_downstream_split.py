import os
import glob
import shutil
import argparse
import random

def setup_args():
    parser = argparse.ArgumentParser(description="Deterministic patient-level train/test split for the downstream classifier benchmark")
    parser.add_argument("--positives-dir", type=str, default="./output/phase1_data", help="Real positive (ct+mask) patches")
    parser.add_argument("--negatives-dir", type=str, default="./output/downstream/negatives", help="Real negative (ct only) patches")
    parser.add_argument("--out-dir", type=str, default="./output/downstream/split")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def uid_of(path, prefix_parts=2):
    """patch_XXXX_<seriesuid>_ct.npy -> <seriesuid>. seriesuid itself has no underscores (dot-separated)."""
    base = os.path.basename(path)
    for suffix in ("_ct.npy", "_mask.npy"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    parts = base.split("_")
    return "_".join(parts[prefix_parts:]) if len(parts) > prefix_parts else base

def copy_positive(ct_path, dest_dir):
    mask_path = ct_path.replace("_ct.npy", "_mask.npy")
    shutil.copy(ct_path, os.path.join(dest_dir, os.path.basename(ct_path)))
    shutil.copy(mask_path, os.path.join(dest_dir, os.path.basename(mask_path)))

def main():
    args = setup_args()

    dirs = {
        "real_train_pos": os.path.join(args.out_dir, "real_train_pos"),
        "real_test_pos": os.path.join(args.out_dir, "real_test_pos"),
        "real_train_neg": os.path.join(args.out_dir, "real_train_neg"),
        "real_test_neg": os.path.join(args.out_dir, "real_test_neg"),
    }
    for d in dirs.values():
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    pos_files = sorted(glob.glob(os.path.join(args.positives_dir, "*_ct.npy")))
    # negatives are named negative_XXXX_<seriesuid>_ct.npy -> same prefix_parts=2 scheme
    neg_files = sorted(glob.glob(os.path.join(args.negatives_dir, "*_ct.npy")))

    # Unified patient-level (seriesuid) split: a scan is ENTIRELY train or ENTIRELY test,
    # regardless of whether it contributes positive or negative patches. This prevents any
    # partial-scan leakage into generator training.
    all_uids = sorted(set(uid_of(f) for f in pos_files) | set(uid_of(f) for f in neg_files))
    rng = random.Random(args.seed)
    shuffled_uids = list(all_uids)
    rng.shuffle(shuffled_uids)
    n_test_uids = int(len(shuffled_uids) * args.test_frac)
    test_uid_set = set(shuffled_uids[:n_test_uids])
    train_uid_set = set(shuffled_uids[n_test_uids:])
    print(f"Unique scans (seriesuid): {len(all_uids)} total -> {len(train_uid_set)} train / {len(test_uid_set)} test")

    pos_train = [f for f in pos_files if uid_of(f) in train_uid_set]
    pos_test = [f for f in pos_files if uid_of(f) in test_uid_set]
    print(f"Positives: {len(pos_files)} total -> {len(pos_train)} train / {len(pos_test)} test")
    for f in pos_train:
        copy_positive(f, dirs["real_train_pos"])
    for f in pos_test:
        copy_positive(f, dirs["real_test_pos"])

    neg_train = [f for f in neg_files if uid_of(f) in train_uid_set]
    neg_test = [f for f in neg_files if uid_of(f) in test_uid_set]
    print(f"Negatives: {len(neg_files)} total -> {len(neg_train)} train / {len(neg_test)} test")
    for f in neg_train:
        shutil.copy(f, os.path.join(dirs["real_train_neg"], os.path.basename(f)))
    for f in neg_test:
        shutil.copy(f, os.path.join(dirs["real_test_neg"], os.path.basename(f)))

    # Sanity check: zero seriesuid overlap between train and test, across both classes
    overlap = train_uid_set & test_uid_set
    if overlap:
        print(f"WARNING: {len(overlap)} seriesuid(s) appear in both train and test!")
    else:
        print("Verified: zero seriesuid overlap between train/test (patient-level split, positives and negatives share one assignment).")

    print(f"\nDone! Split saved under {args.out_dir}")
    for name, d in dirs.items():
        print(f"  {name}: {d} ({len(os.listdir(d))} files)")

if __name__ == "__main__":
    main()
