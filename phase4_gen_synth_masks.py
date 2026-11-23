import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import label
from tqdm import tqdm

# 1. Architecture
class ResBlock3D(nn.Module):
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
        # The bottleneck stays at bottleneck_spatial^3; one stride-2 stage is added per
        # doubling of input_size (channels capped at max_channels).
        super(VAE3D, self).__init__()

        num_stages = round(np.log2(input_size / bottleneck_spatial))
        assert bottleneck_spatial * (2 ** num_stages) == input_size, \
            f"input_size={input_size} must be bottleneck_spatial({bottleneck_spatial}) * a power of 2"

        channel_list = [32]
        for _ in range(num_stages):
            channel_list.append(min(channel_list[-1] * 2, max_channels))

        self.channel_list = channel_list
        self.bottleneck_spatial = bottleneck_spatial
        self.bottleneck_channels = channel_list[-1]
        bottleneck_flat = self.bottleneck_channels * (bottleneck_spatial ** 3)

        encoder_layers = [
            nn.Conv3d(1, channel_list[0], kernel_size=3, stride=1, padding=1),
            ResBlock3D(channel_list[0]),
        ]
        for i in range(1, len(channel_list)):
            encoder_layers.append(nn.Conv3d(channel_list[i - 1], channel_list[i], kernel_size=3, stride=2, padding=1))
            encoder_layers.append(ResBlock3D(channel_list[i]))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(bottleneck_flat, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, bottleneck_flat)

        decoder_layers = []
        for i in range(len(channel_list) - 1, 0, -1):
            decoder_layers.append(ResBlock3D(channel_list[i]))
            decoder_layers.append(nn.ConvTranspose3d(channel_list[i], channel_list[i - 1], kernel_size=4, stride=2, padding=1))
        decoder_layers.append(ResBlock3D(channel_list[0]))
        decoder_layers.append(nn.Conv3d(channel_list[0], 1, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.bottleneck_channels, self.bottleneck_spatial, self.bottleneck_spatial, self.bottleneck_spatial)
        return self.decoder(h)

# Connected-component filter
def extract_main_vascular_tree(binary_mask, min_volume):
    """
    Keeps all connected components that have a voxel count >= min_volume.
    """
    labels = label(binary_mask, connectivity=3)
    
    if labels.max() == 0:
        return binary_mask
        
    component_sizes = np.bincount(labels.flat)[1:]
    
    if min_volume > 0:
        # Find all label IDs that meet the volume threshold
        valid_labels = np.where(component_sizes >= min_volume)[0] + 1
        clean_mask = np.isin(labels, valid_labels).astype(np.float32)
    else:
        # Fallback to keeping only the single absolute largest if thresh is 0
        largest_component_id = np.argmax(component_sizes) + 1
        clean_mask = (labels == largest_component_id).astype(np.float32)
        
    return clean_mask

# Mass Generation Logic
def setup_args():
    parser = argparse.ArgumentParser(description="Phase 4: Generate Synthetic Dataset")
    parser.add_argument("--model-path", type=str, default="./output/models/vae_geometry_heavy_best.pth")
    parser.add_argument("--out-dir", type=str, default="./output/phase4_synthetic_data")
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5, help="VAE probability threshold")
    parser.add_argument("--cca-thresh", type=int, default=75, help="Minimum voxel volume to keep a vessel branch")
    parser.add_argument("--input-size", type=int, default=32, help="Cubic patch size the VAE was trained on")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Could not find model at {args.model_path}")
        return

    print(f"Loading VAE from {args.model_path}...")
    model = VAE3D(latent_dim=args.latent_dim, input_size=args.input_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(args.num_samples), desc="Synthesizing Dataset"):
            z = torch.randn(1, args.latent_dim).to(device)
            raw_probs = model.decode(z).squeeze().cpu().numpy()
            
            binary_mask = (raw_probs > args.threshold).astype(np.float32)
            clean_tree = extract_main_vascular_tree(binary_mask, args.cca_thresh)
            
            out_filename = os.path.join(args.out_dir, f"synthetic_vessel_{i:04d}_mask.npy")
            np.save(out_filename, clean_tree)
            
    print(f"\nDone! Saved {args.num_samples} masks to {args.out_dir}")

if __name__ == "__main__":
    main()