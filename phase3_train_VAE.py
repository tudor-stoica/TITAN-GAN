import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 3D ResNet VAE Architecture
# ==========================================
class ResBlock3D(nn.Module):
    """3D residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class VAE3D(nn.Module):
    def __init__(self, latent_dim=512, input_size=32, bottleneck_spatial=4, max_channels=256):
        # Increased latent dimension for richer geometry
        # The bottleneck stays at bottleneck_spatial^3; one stride-2 stage is added per
        # doubling of input_size (channels capped at max_channels).
        super(VAE3D, self).__init__()

        num_stages = round(np.log2(input_size / bottleneck_spatial))
        assert bottleneck_spatial * (2 ** num_stages) == input_size, \
            f"input_size={input_size} must be bottleneck_spatial({bottleneck_spatial}) * a power of 2"

        # channel_list[0] is the initial (stride-1) conv output; channel_list[1:] are the stride-2 stages
        channel_list = [32]
        for _ in range(num_stages):
            channel_list.append(min(channel_list[-1] * 2, max_channels))

        self.channel_list = channel_list
        self.bottleneck_spatial = bottleneck_spatial
        self.bottleneck_channels = channel_list[-1]
        bottleneck_flat = self.bottleneck_channels * (bottleneck_spatial ** 3)

        # Encoder: input_size -> ... -> bottleneck_spatial
        encoder_layers = [
            nn.Conv3d(1, channel_list[0], kernel_size=3, stride=1, padding=1),
            ResBlock3D(channel_list[0]),
        ]
        for i in range(1, len(channel_list)):
            encoder_layers.append(nn.Conv3d(channel_list[i - 1], channel_list[i], kernel_size=3, stride=2, padding=1))
            encoder_layers.append(ResBlock3D(channel_list[i]))
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent Space
        self.fc_mu = nn.Linear(bottleneck_flat, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_flat, latent_dim)

        # Decoder Projection
        self.fc_decode = nn.Linear(latent_dim, bottleneck_flat)

        # Decoder: bottleneck_spatial -> ... -> input_size
        decoder_layers = []
        for i in range(len(channel_list) - 1, 0, -1):
            decoder_layers.append(ResBlock3D(channel_list[i]))
            decoder_layers.append(nn.ConvTranspose3d(channel_list[i], channel_list[i - 1], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(ResBlock3D(channel_list[0]))
        decoder_layers.append(nn.Conv3d(channel_list[0], 1, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.bottleneck_channels, self.bottleneck_spatial, self.bottleneck_spatial, self.bottleneck_spatial)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ==========================================
# 2. Dataset Loader 
# ==========================================
class VaeMaskDataset(Dataset):
    def __init__(self, data_dir, augment=True):
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, "*_mask.npy")))
        self.augment = augment
        
    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_patch = np.load(self.mask_files[idx])
        
        if self.augment:
            # Randomly flip across Z, Y, or X axes
            if np.random.random() > 0.5: mask_patch = np.flip(mask_patch, axis=0)
            if np.random.random() > 0.5: mask_patch = np.flip(mask_patch, axis=1)
            if np.random.random() > 0.5: mask_patch = np.flip(mask_patch, axis=2)
            
            # Random 90-degree rotations (k = number of 90-deg turns)
            # This works on any 2 axes (e.g., rotating in the XY plane)
            k = np.random.randint(0, 4)
            mask_patch = np.rot90(mask_patch, k=k, axes=(1, 2))
            
        mask_tensor = torch.from_numpy(mask_patch.copy()).unsqueeze(0).float()
        return mask_tensor

# Topology-Aware Loss Function
def soft_dice_loss(pred, target, smooth=1e-5):
    """Soft-Dice loss between predicted and target masks."""
    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def vae_loss_function(recon_x, x, mu, logvar, beta=0.1, dice_weight=10.0):
    # Pixel-wise precision
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    
    # Soft-Dice overlap term
    DICE = soft_dice_loss(recon_x, x)
    
    # Latent space organization
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = BCE + (dice_weight * DICE) + (beta * KLD)
    return total_loss, BCE, DICE, KLD

# Training Loop
def setup_args():
    parser = argparse.ArgumentParser(description="Phase 3: Train the Heavy 3D Geometry VAE")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--out-dir", type=str, default="./output/models")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly lower LR for deep ResNet stability
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dice-weight", type=float, default=7.0, help="Weight to enforce connectivity")
    parser.add_argument("--input-size", type=int, default=32, help="Cubic patch size (must be 4 * a power of 2)")
    parser.add_argument("--latent-dim", type=int, default=512)
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Heavy Phase 3 VAE on: {device}")
    
    dataset = VaeMaskDataset(args.data_dir)
    if len(dataset) == 0:
        print("Error: No mask data found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = VAE3D(latent_dim=args.latent_dim, input_size=args.input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    history = {'total': [], 'bce': [], 'dice': [], 'kld': []}
    best_loss = float('inf')
    best_model_path = os.path.join(args.out_dir, "vae_geometry_heavy_best.pth")
    
    print("Starting VAE training loop...")
    
    # The Global Epoch Progress Bar
    global_loop = tqdm(range(1, args.epochs + 1), desc="Overall Progress", unit="epoch")
    
    for epoch in global_loop:
        model.train()
        epoch_total, epoch_bce, epoch_dice, epoch_kld = 0.0, 0.0, 0.0, 0.0
        
        # The Inner Batch Progress Bar (leave=False makes it disappear when the epoch is done)
        batch_loop = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{args.epochs}]", leave=False)
        
        for mask_b in batch_loop:
            mask_b = mask_b.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(mask_b)
            loss, bce, dice, kld = vae_loss_function(recon_batch, mask_b, mu, logvar, args.beta, args.dice_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_total += loss.item()
            epoch_bce += bce.item()
            epoch_dice += dice.item()
            epoch_kld += kld.item()
            
            # Update inner bar with live loss
            batch_loop.set_postfix(loss=loss.item())
            
        batches = len(dataloader)
        avg_total = epoch_total / batches
        avg_bce = epoch_bce / batches
        avg_dice = epoch_dice / batches
        avg_kld = epoch_kld / batches
        
        history['total'].append(avg_total)
        history['bce'].append(avg_bce)
        history['dice'].append(avg_dice)
        history['kld'].append(avg_kld)
        
        saved_flag = ""
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), best_model_path)
            saved_flag = "-> [NEW BEST SAVED]"

        if epoch % 100 == 0:
            checkpoint_path = os.path.join(args.out_dir, f"vae_heavy_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            global_loop.write(f"--> [INTERMEDIATE CHECKPOINT SAVED: {checkpoint_path}]")

        scheduler.step(avg_total)
            
        # Cleanly print the epoch summary ABOVE the progress bars
        global_loop.write(f"Epoch [{epoch:03d}/{args.epochs}] | Total: {avg_total:.3f} | BCE: {avg_bce:.3f} | DiceLoss: {avg_dice:.3f} | KLD: {avg_kld:.3f} {saved_flag}")

    print("Generating VAE metrics plot...")
    epochs_range = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Reconstruction (BCE + Dice)', color='blue', fontweight='bold')
    ax1.plot(epochs_range, history['bce'], label='BCE Loss', color='blue', linewidth=2, alpha=0.5)
    ax1.plot(epochs_range, history['dice'], label='Dice Loss', color='navy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('KL Divergence', color='red', fontweight='bold')
    ax2.plot(epochs_range, history['kld'], label='KLD Loss', color='red', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title('Heavy 3D ResNet VAE Training Metrics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "vae_heavy_training_metrics.png"), dpi=150)
    
    print("Testing Latent Space...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, args.latent_dim).to(device)
        novel_mask = model.decode(z).squeeze().cpu().numpy()
        novel_mask_binary = novel_mask > 0.5
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(novel_mask_binary, facecolors='green', edgecolor='k', linewidth=0.1, alpha=0.8)
        ax.set_axis_off()
        plt.title("ResNet VAE Generated Vessel Topology")
        plt.savefig(os.path.join(args.out_dir, "novel_heavy_topology.png"), dpi=150)
        
    print(f"Done! Saved weights, plots, and a sample topology to {args.out_dir}")

if __name__ == "__main__":
    main()