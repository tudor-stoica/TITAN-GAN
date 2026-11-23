import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil

# Generator Architecture
class UNet3D_Generator(nn.Module):
    """The Heavy 3D U-Net to render CT tissue from Vessel Masks"""
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        
        self.down1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bottleneck = self._block(features * 4, features * 8)
        
        self.up1 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.up_conv1 = self._block(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.up_conv2 = self._block(features * 4, features * 2)
        
        self.up3 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.up_conv3 = self._block(features * 2, features)
        
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

# Generation Logic
def setup_args():
    parser = argparse.ArgumentParser(description="Phase 7: Mint Final Synthetic Dataset")
    parser.add_argument("--mask-dir", type=str, default="./output/phase4_synthetic_data", help="Folder with VAE masks")
    parser.add_argument("--gen-path", type=str, default="./output/models/tighterTitan/cgan_gen_epoch_350.pth", help="Best GAN checkpoint")
    parser.add_argument("--out-dir", type=str, default="./output/phase7_final_dataset_NEW", help="Where to save the paired dataset")
    parser.add_argument("--convert-hu", action="store_true", help="Flag to convert data back to Hounsfield Units (-1000 to 400)")
    return parser.parse_args()

def main():
    args = setup_args()
    
    # Create a fresh directory for the final dataset
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the Best Generator
    if not os.path.exists(args.gen_path):
        print(f"Error: Generator not found at {args.gen_path}")
        return
        
    print(f"Loading Master Generator: {args.gen_path}")
    generator = UNet3D_Generator().to(device)
    generator.load_state_dict(torch.load(args.gen_path, map_location=device))
    generator.eval()
    
    # Find Phase 4 Synthetic Masks
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*_mask.npy")))
    num_files = len(mask_files)
    
    if num_files == 0:
        print(f"Error: No synthetic masks found in {args.mask_dir}")
        return
        
    print(f"Discovered {num_files} synthetic blueprints. Commencing generation...")
    
    # Mint the Dataset
    with torch.no_grad():
        for mask_path in tqdm(mask_files, desc="Rendering Final CT Volumes"):
            # Load the blueprint mask
            mask_np = np.load(mask_path)
            
            # Format for PyTorch (Batch=1, Channel=1, Z, Y, X)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Hallucinate the CT tissue
            fake_ct_tensor = generator(mask_tensor)
            
            # Convert back to numpy
            fake_ct_np = fake_ct_tensor.squeeze().cpu().numpy()
            
            # Convert [0, 1] neural network range back to clinical Hounsfield Units
            if args.convert_hu:
                fake_ct_np = (fake_ct_np * 1400.0) - 1000.0
                fake_ct_np = fake_ct_np.astype(np.int16) # HU are usually integers
            
            # Save the Pair (Mask and CT) to the final folder
            base_name = os.path.basename(mask_path).replace('_mask.npy', '')
            
            # Copy the mask over so the dataset is complete and paired
            out_mask_path = os.path.join(args.out_dir, f"{base_name}_mask.npy")
            shutil.copy(mask_path, out_mask_path)
            
            # Save the new CT
            out_ct_path = os.path.join(args.out_dir, f"{base_name}_ct.npy")
            np.save(out_ct_path, fake_ct_np)

    print("\n" + "="*50)
    print(" PIPELINE COMPLETE ")
    print("="*50)
    print(f"Successfully generated {num_files} paired clinical patches.")
    print(f"Final dataset is secured at: {args.out_dir}")

if __name__ == "__main__":
    main()