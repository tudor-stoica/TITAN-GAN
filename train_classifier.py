import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from classifier_3d import Classifier3D, NoduleDataset, compute_metrics

def setup_args():
    parser = argparse.ArgumentParser(description="Train/evaluate one downstream classifier condition")
    parser.add_argument("--condition-name", type=str, required=True, help="e.g. real_only, vanilla, alpha_wgan, titan")
    parser.add_argument("--real-train-pos-dir", type=str, default="./output/downstream/split/real_train_pos")
    parser.add_argument("--real-train-neg-dir", type=str, default="./output/downstream/split/real_train_neg")
    parser.add_argument("--synth-pos-dir", type=str, default=None, help="Optional synthetic positive augmentation dir")
    parser.add_argument("--real-test-pos-dir", type=str, default="./output/downstream/split/real_test_pos")
    parser.add_argument("--real-test-neg-dir", type=str, default="./output/downstream/split/real_test_neg")
    parser.add_argument("--out-dir", type=str, default="./output/downstream/classifier_runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Carved out of TRAINING data only, for checkpoint selection - never the locked test set")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for ct, label in loader:
            ct, label = ct.to(device), label.to(device)
            logits = model(ct).squeeze(1)
            loss = criterion(logits, label)
            total_loss += loss.item() * ct.size(0)
            probs = torch.sigmoid(logits)
            all_labels.extend(label.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    avg_loss = total_loss / max(1, len(loader.dataset))
    return avg_loss, all_labels, all_probs

def main():
    args = setup_args()
    run_dir = os.path.join(args.out_dir, args.condition_name)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.condition_name}] Initializing on: {device}")

    # Two dataset instances (aug on/off) sharing the same deterministic file ordering,
    # so index-based train/val split lines up between them.
    real_pos_only = NoduleDataset(args.real_train_pos_dir, args.real_train_neg_dir, synth_pos_dir=None, augment=False)
    n_real_pos = real_pos_only.n_pos
    full_noaug = NoduleDataset(args.real_train_pos_dir, args.real_train_neg_dir, args.synth_pos_dir, augment=False)
    full_aug = NoduleDataset(args.real_train_pos_dir, args.real_train_neg_dir, args.synth_pos_dir, augment=True)
    n = len(full_noaug)
    n_synth = full_noaug.n_pos - n_real_pos
    print(f"[{args.condition_name}] Training pool: {n_real_pos} real positive + {n_synth} synthetic positive "
          f"= {full_noaug.n_pos} total positive / {full_noaug.n_neg} negative, {n} total")

    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    n_val = int(n * args.val_frac)
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    train_loader = DataLoader(Subset(full_aug, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_noaug, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_dataset = NoduleDataset(args.real_test_pos_dir, args.real_test_neg_dir, synth_pos_dir=None, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[{args.condition_name}] Train: {len(train_indices)} | Val (held out from train only): {len(val_indices)} | Locked real test: {test_dataset.n_pos} pos / {test_dataset.n_neg} neg")

    model = Classifier3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_ckpt_path = os.path.join(run_dir, "classifier_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for ct, label in train_loader:
            ct, label = ct.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(ct).squeeze(1)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ct.size(0)
        avg_train_loss = train_loss / max(1, len(train_loader.dataset))

        val_loss, val_labels, val_probs = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs)

        saved_flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            saved_flag = "-> [NEW BEST]"

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{args.condition_name}] Epoch [{epoch:03d}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | Val Sens: {val_metrics['Sensitivity']:.4f} {saved_flag}")

    # Final evaluation on the locked real test set, using the best (val-selected) checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_labels, test_probs = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs)
    test_metrics["condition"] = args.condition_name
    test_metrics["test_loss"] = test_loss
    test_metrics["n_synth_augment"] = n_synth
    test_metrics["n_real_train_pos"] = n_real_pos

    print(f"\n[{args.condition_name}] === FINAL RESULTS on locked real test set ===")
    for k in ["Accuracy", "Sensitivity", "Specificity", "Precision", "AUC"]:
        print(f"  {k}: {test_metrics[k]:.4f}")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nDone! Results saved to {run_dir}")

if __name__ == "__main__":
    main()
