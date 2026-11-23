import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
from tqdm import tqdm

# The Generator (Heavy 3D U-Net)
class UNet3D_Generator(nn.Module):
    """Translates a Binary Vessel Mask into a 3D CT Patch"""
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        
        # Downsampling
        self.down1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)
        
        # Upsampling
        self.up1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.up_conv1 = self._block(features * 8, features * 4) # *8 because of concat
        
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.up_conv2 = self._block(features * 4, features * 2)
        
        self.up3 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.up_conv3 = self._block(features * 2, features)
        
        # Final Output
        self.final = nn.Sequential(
            nn.Conv3d(features, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        
        b = self.bottleneck(self.pool3(d3))
        
        u1 = self.up_conv1(torch.cat([self.up1(b), d3], dim=1))
        u2 = self.up_conv2(torch.cat([self.up2(u1), d2], dim=1))
        u3 = self.up_conv3(torch.cat([self.up3(u2), d1], dim=1))
        
        return self.final(u3)

# The Discriminator (3D PatchGAN)
class PatchGAN3D_Discriminator(nn.Module):
    """Evaluates local 3D textures with Spectral Normalization for stability."""
    def __init__(self, in_channels=2, features=64):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv3d(features, features * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv3d(features * 2, features * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv3d(features * 4, 1, kernel_size=4, stride=1, padding=1))
        )

    def forward(self, mask, ct):
        x = torch.cat([mask, ct], dim=1)
        return self.model(x)

# Topology Loss Enforcer (Frozen Phase 2)
class SoftGraphBridge(nn.Module):
    """The exact architecture from Phase 2"""
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

class FrozenPhase2CNN(nn.Module):
    """Loads the trained Soft-Bridge (phase 2) and freezes it for use as a loss."""
    def __init__(self, phase2_weights_path, device):
        super().__init__()
        
        self.model = SoftGraphBridge()
        self.model.load_state_dict(torch.load(phase2_weights_path, map_location=device))
        self.model.to(device)
        self.model.eval() # Strictly evaluation mode
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, fake_ct):
        return self.model(fake_ct)

def soft_dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = pred.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

# Paired Dataset Loader (Phase 1 Data)
class PairedCTDataset(Dataset):
    def __init__(self, data_dir):
        # Grab all masks, then find the corresponding CT files
        self.mask_files = sorted(glob.glob(os.path.join(data_dir, "*_mask.npy")))
        self.ct_files = [f.replace('_mask.npy', '_ct.npy') for f in self.mask_files]
        
    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask = np.load(self.mask_files[idx])
        ct = np.load(self.ct_files[idx])
        
        # Paired Augmentation: Randomly flip along X, Y, Z axes
        if np.random.rand() > 0.5:
            mask = np.flip(mask, axis=0)
            ct = np.flip(ct, axis=0)
        if np.random.rand() > 0.5:
            mask = np.flip(mask, axis=1)
            ct = np.flip(ct, axis=1)
        if np.random.rand() > 0.5:
            mask = np.flip(mask, axis=2)
            ct = np.flip(ct, axis=2)
            
        # Paired Augmentation: Random 90-degree rotations in the XY plane
        k_rot = np.random.randint(0, 4)
        if k_rot > 0:
            mask = np.rot90(mask, k=k_rot, axes=(0, 1))
            ct = np.rot90(ct, k=k_rot, axes=(0, 1))
        
        # Note: .copy() is required because np.flip returns a negative stride array
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()
        ct_tensor = torch.from_numpy(ct.copy()).unsqueeze(0).float()
        
        return mask_tensor, ct_tensor

# Training Loop
def setup_args():
    parser = argparse.ArgumentParser(description="Phase 5: Train the 3D Conditional GAN")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data", help="Directory with Paired Real Phase 1 Data")
    parser.add_argument("--phase2-path", type=str, default="./output/models/soft_graph_bridge_best.pth", help="Path to frozen Phase 2 CNN")
    parser.add_argument("--out-dir", type=str, default="./output/models/tighterTitan")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8) # GANs require smaller batches due to memory constraints
    parser.add_argument("--lr-g", type=float, default=2e-4, help="Generator Learning Rate")
    parser.add_argument("--lr-d", type=float, default=5e-5, help="Discriminator Learning Rate (Slower)")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="Weight for Reconstruction Loss")
    parser.add_argument("--lambda-topo", type=float, default=15.0, help="Weight for clDice Topology Loss")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Heavy 3D cGAN on: {device}")
    
    # Load Data
    dataset = PairedCTDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Initialize Networks
    generator = UNet3D_Generator().to(device)
    discriminator = PatchGAN3D_Discriminator().to(device)
    topology_enforcer = FrozenPhase2CNN(args.phase2_path, device)
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.999))
    
    # Loss Functions
    criterion_GAN = nn.MSELoss() # LSGAN is more stable than BCE for PatchGAN
    criterion_L1 = nn.L1Loss()
    
    print("Starting cGAN training loop...")
    global_loop = tqdm(range(1, args.epochs + 1), desc="Overall Progress", unit="epoch")

    history_D = []
    history_G = []
    
    for epoch in global_loop:
        generator.train()
        discriminator.train()
        epoch_d_loss, epoch_g_loss = 0.0, 0.0
        
        batch_loop = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{args.epochs}]", leave=False)
        
        for real_mask, real_ct in batch_loop:
            real_mask = real_mask.to(device)
            real_ct = real_ct.to(device)
            
            # Train Discriminator
            opt_D.zero_grad()
            
            # Generate Fake CT
            fake_ct = generator(real_mask)
            
            # Real Patch Loss
            pred_real = discriminator(real_mask, real_ct)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake Patch Loss (detach fake_ct so we don't backprop through G)
            pred_fake = discriminator(real_mask, fake_ct.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            # Total D Loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            
            # Adversarial Loss (Did we fool D?)
            pred_fake_for_G = discriminator(real_mask, fake_ct)
            loss_G_adv = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            
            # Reconstruction Loss (Does it look exactly like the real CT?)
            loss_G_L1 = criterion_L1(fake_ct, real_ct)
            
            # Topology Loss (Did we break the vessels?)

            # Pass the hallucinated CT through Phase 2 to get the "Hallucinated Mask"
            fake_mask_prediction = topology_enforcer(fake_ct)
            # Compare Hallucinated Mask to Real Input Mask
            loss_G_topo = soft_dice_loss(fake_mask_prediction, real_mask)
            
            # Total G Loss
            loss_G = loss_G_adv + (args.lambda_l1 * loss_G_L1) + (args.lambda_topo * loss_G_topo)
            loss_G.backward()
            opt_G.step()
            
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            
            batch_loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())
            
        # Logging & Checkpointing
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        global_loop.write(f"Epoch [{epoch:03d}/{args.epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

        history_D.append(avg_d_loss)
        history_G.append(avg_g_loss)
        
        # Save checkpoints
        if epoch % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(args.out_dir, f"cgan_gen_epoch_{epoch:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.out_dir, f"cgan_disc_epoch_{epoch:03d}.pth"))
            global_loop.write(f"--> [CHECKPOINT SAVED]")


    # Plotting
    print("Generating GAN performance plot...")
    
    epochs_range = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot G Loss
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Generator Loss (L1 + Adv + Topo)', color='blue', fontweight='bold')
    ax1.plot(epochs_range, history_G, label='Generator Loss', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Plot D Loss 
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Discriminator Loss', color='red', fontweight='bold')
    ax2.plot(epochs_range, history_D, label='Discriminator Loss', color='red', linewidth=2, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, max(history_D) * 2]) # Keep D loss scaled nicely
    ax2.legend(loc='upper right')

    plt.title('Conditional GAN Training Equilibrium', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.out_dir, "cgan_training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()