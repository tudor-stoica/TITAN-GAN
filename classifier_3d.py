import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.stats import rankdata

# ==========================================
# 3D Nodule/Non-Nodule Classifier
# Small 4-block 3D CNN used for the downstream augmentation experiments.
# ==========================================
class Classifier3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool3d(2),   # 32 -> 16

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool3d(2),   # 16 -> 8

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.2, inplace=True), nn.MaxPool3d(2),   # 8 -> 4

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)  # raw logits, use BCEWithLogitsLoss

# ==========================================
# Dataset: real positives (+ optional synthetic positives) vs real negatives
# ==========================================
class NoduleDataset(Dataset):
    def __init__(self, real_pos_dir, real_neg_dir, synth_pos_dir=None, augment=True):
        pos_files = sorted(glob.glob(os.path.join(real_pos_dir, "*_ct.npy")))
        if synth_pos_dir:
            # Synthetic dirs use different file names (*_ct.npy, vanilla_ct_*.npy,
            # alpha_wgan_ct_*.npy) and some also contain *_mask.npy files, which are skipped.
            synth_all = sorted(glob.glob(os.path.join(synth_pos_dir, "*.npy")))
            pos_files += [f for f in synth_all if not f.endswith("_mask.npy")]
        neg_files = sorted(glob.glob(os.path.join(real_neg_dir, "*_ct.npy")))

        self.samples = [(f, 1.0) for f in pos_files] + [(f, 0.0) for f in neg_files]
        self.augment = augment
        self.n_pos = len(pos_files)
        self.n_neg = len(neg_files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        ct = np.load(path)

        if self.augment:
            if np.random.rand() > 0.5: ct = np.flip(ct, axis=0)
            if np.random.rand() > 0.5: ct = np.flip(ct, axis=1)
            if np.random.rand() > 0.5: ct = np.flip(ct, axis=2)
            k = np.random.randint(0, 4)
            if k > 0: ct = np.rot90(ct, k=k, axes=(1, 2))

        ct_tensor = torch.from_numpy(ct.copy()).unsqueeze(0).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return ct_tensor, label_tensor

# ==========================================
# Metrics. AUC is computed with the Mann-Whitney U / rank-sum identity
# (scipy.stats.rankdata), so sklearn is not required.
# ==========================================
def compute_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(np.float32)

    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))

    accuracy = (tp + tn) / max(1.0, (tp + tn + fp + fn))
    sensitivity = tp / max(1.0, (tp + fn))  # recall
    specificity = tn / max(1.0, (tn + fp))
    precision = tp / max(1.0, (tp + fp))

    n_pos, n_neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if n_pos > 0 and n_neg > 0:
        ranks = rankdata(probs)
        sum_ranks_pos = ranks[labels == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    else:
        auc = float('nan')

    return {
        "Accuracy": accuracy, "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "AUC": auc, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "N_pos": n_pos, "N_neg": n_neg,
    }
