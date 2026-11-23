"""
Rebuild the patient-level downstream split used for the paper's classification results.

The split is the seed-42 patient-level split from make_downstream_split.py computed over
the positives and the random negatives (output/downstream/negatives). The train-only
generators were trained on its training positives. The hard negatives
(output/downstream/negatives_hard) are then assigned by patient: a hard negative goes to
the test set if and only if its scan is a test scan.

Result: 481 train / 134 test positives, with no scan in both train and test.
Files are symlinked rather than copied.
"""
import os, sys, glob, random, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from make_downstream_split import uid_of

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=f"{ROOT}/output/downstream/split_orig")
args = ap.parse_args()

pos = sorted(glob.glob(f"{ROOT}/output/phase1_data/*_ct.npy"))
easy_neg = sorted(glob.glob(f"{ROOT}/output/downstream/negatives/*_ct.npy"))
hard_neg = sorted(glob.glob(f"{ROOT}/output/downstream/negatives_hard/*_ct.npy"))

uids = sorted(set(map(uid_of, pos)) | set(map(uid_of, easy_neg)))
shuffled = list(uids)
random.Random(42).shuffle(shuffled)
test_uids = set(shuffled[: int(len(shuffled) * 0.2)])

dirs = {k: os.path.join(args.out_dir, k) for k in ["real_train_pos", "real_test_pos", "real_train_neg", "real_test_neg"]}
for d in dirs.values():
    if os.path.exists(d) and os.listdir(d):
        sys.exit(f"{d} already exists and is non-empty; refusing to overwrite")
    os.makedirs(d, exist_ok=True)

def link(src, dst_dir):
    os.symlink(src, os.path.join(dst_dir, os.path.basename(src)))

n = {k: 0 for k in dirs}
for p in pos:
    k = "real_test_pos" if uid_of(p) in test_uids else "real_train_pos"
    link(p, dirs[k]); link(p.replace("_ct.npy", "_mask.npy"), dirs[k]); n[k] += 1
for p in hard_neg:
    k = "real_test_neg" if uid_of(p) in test_uids else "real_train_neg"
    link(p, dirs[k]); n[k] += 1

train_scans = {uid_of(p) for d in ["real_train_pos", "real_train_neg"] for p in glob.glob(os.path.join(dirs[d], "*_ct.npy"))}
test_scans = {uid_of(p) for d in ["real_test_pos", "real_test_neg"] for p in glob.glob(os.path.join(dirs[d], "*_ct.npy"))}
overlap = train_scans & test_scans
print(n)
print(f"scan overlap train/test: {len(overlap)}")
assert not overlap and n["real_test_pos"] == 134 and n["real_train_pos"] == 481
print(f"OK: original split + hard negatives written to {args.out_dir}")
