import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Architecture
class SoftGraphBridge(nn.Module):
    def __init__(self):
        super(SoftGraphBridge, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

# Dataset Loader
class LunaPatchDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ct_files = sorted(glob.glob(os.path.join(data_dir, "*_ct.npy")))
        
    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_path = self.ct_files[idx]
        mask_path = ct_path.replace("_ct.npy", "_mask.npy")
        
        ct_tensor = torch.from_numpy(np.load(ct_path)).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(np.load(mask_path)).unsqueeze(0).float()
        
        return ct_tensor, mask_tensor

# Metrics
def calculate_dice(preds, targets, smooth=1e-5):
    """Calculates the Dice overlap. Thresholds the soft predictions to 0.5 first."""
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# Training Loop
def setup_args():
    parser = argparse.ArgumentParser(description="Phase 2: Train the Soft-Graph Bridge")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--out-dir", type=str, default="./output/models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Phase 2 Training on: {device}")
    
    # Data Splitting (80/20)
    full_dataset = LunaPatchDataset(args.data_dir)
    if len(full_dataset) == 0:
        print("Error: No data found.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data Split: {train_size} Train | {val_size} Validation")
    
    # Model Setup
    model = SoftGraphBridge().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Tracking
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.out_dir, "soft_graph_bridge_best.pth")
    
    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for ct_b, mask_b in train_loader:
            ct_b, mask_b = ct_b.to(device), mask_b.to(device)
            optimizer.zero_grad()
            outputs = model(ct_b)
            loss = criterion(outputs, mask_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for ct_b, mask_b in val_loader:
                ct_b, mask_b = ct_b.to(device), mask_b.to(device)
                outputs = model(ct_b)
                val_loss += criterion(outputs, mask_b).item()
                val_dice += calculate_dice(outputs, mask_b).item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)
        
        # 3. Model Checkpointing
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            saved_flag = "-> [NEW BEST SAVED]"
            
        print(f"Epoch [{epoch:02d}/{args.epochs}] | Train BCE: {avg_train_loss:.4f} | Val BCE: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} {saved_flag}")

    # Plotting
    print("Generating performance plots...")
    epochs_range = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Losses on left Y-axis
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('BCE Loss', color='black', fontweight='bold')
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='red', linewidth=2, linestyle='--')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # Plot Dice on right Y-axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Dice Score', color='green', fontweight='bold')
    ax2.plot(epochs_range, history['val_dice'], label='Val Dice', color='green', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1.0]) # Dice is strictly bounded 0.0 to 1.0
    ax2.legend(loc='upper right')

    plt.title('Soft-Graph Bridge Training Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.out_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()