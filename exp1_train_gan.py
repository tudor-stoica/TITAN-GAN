import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader

# WGAN-GP Architectures
class Generator3D(nn.Module):
    """Maps a latent vector to an output_size^3 CT patch. Starts from a 4^3 volume
    and adds one upsampling stage per doubling of output_size."""
    def __init__(self, latent_dim=128, features=64, output_size=32):
        super().__init__()
        self.init_size = 4
        num_stages = round(np.log2(output_size / self.init_size))
        assert self.init_size * (2 ** num_stages) == output_size, \
            f"output_size={output_size} must be init_size({self.init_size}) * a power of 2"

        # Channel count per upsample stage's *input*, halving down to `features` and floored there.
        channel_list = [features * 4]
        for _ in range(num_stages - 1):
            channel_list.append(max(features, channel_list[-1] // 2))

        self.l1 = nn.Sequential(nn.Linear(latent_dim, channel_list[0] * self.init_size**3))

        layers = [nn.BatchNorm3d(channel_list[0])]
        for i in range(num_stages):
            in_ch = channel_list[i]
            is_last = (i == num_stages - 1)
            out_ch = 1 if is_last else channel_list[i + 1]
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            if is_last:
                layers.append(nn.Sigmoid())  # CT data is [0, 1]
            else:
                layers.append(nn.BatchNorm3d(out_ch, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_blocks = nn.Sequential(*layers)
        self.first_channels = channel_list[0]

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.first_channels, self.init_size, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Critic3D(nn.Module):
    """WGAN critic. Downsamples to a fixed 4^3 bottleneck, adding one stride-2
    stage per doubling of input_size."""
    def __init__(self, in_channels=1, features=64, input_size=32):
        super().__init__()
        bottleneck_spatial = 4
        num_stages = round(np.log2(input_size / bottleneck_spatial))
        assert bottleneck_spatial * (2 ** num_stages) == input_size, \
            f"input_size={input_size} must be {bottleneck_spatial} * a power of 2"

        channel_list = [features]
        for _ in range(num_stages - 1):
            channel_list.append(min(features * 4, channel_list[-1] * 2))

        layers = [nn.Conv3d(in_channels, channel_list[0], kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        for i in range(1, len(channel_list)):
            layers.append(nn.Conv3d(channel_list[i - 1], channel_list[i], kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm3d(channel_list[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
        self.adv_layer = nn.Linear(channel_list[-1] * (bottleneck_spatial ** 3), 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

# Dataset & WGAN-GP Logic
class RealCTDataset(Dataset):
    def __init__(self, data_dir):
        self.ct_files = sorted(glob.glob(os.path.join(data_dir, "*_ct.npy")))
        
    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct = np.load(self.ct_files[idx])
        if np.random.rand() > 0.5: ct = np.flip(ct, axis=0)
        if np.random.rand() > 0.5: ct = np.flip(ct, axis=1)
        if np.random.rand() > 0.5: ct = np.flip(ct, axis=2)
        return torch.from_numpy(ct.copy()).unsqueeze(0).float()

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = Variable(torch.ones(real_samples.size(0), 1).to(device), requires_grad=False)
    
    gradients = grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Training Loop
def setup_args():
    parser = argparse.ArgumentParser(description="Exp 1: Train Baseline 3D WGAN-GP")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--out-dir", type=str, default="./output/models/baseline_gan")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=32, help="Cubic patch size")
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator3D(latent_dim=args.latent_dim, output_size=args.patch_size).to(device)
    critic = Critic3D(input_size=args.patch_size).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
    
    dataloader = DataLoader(RealCTDataset(args.data_dir), batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    print(f"Starting WGAN-GP Baseline Training for {args.epochs} Epochs...")
    
    for epoch in range(1, args.epochs + 1):
        for i, real_ct in enumerate(dataloader):
            real_ct = real_ct.to(device)
            batch_size = real_ct.size(0)
            
            # Train Critic
            optimizer_C.zero_grad()
            z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_ct = generator(z).detach()
            
            loss_C = -torch.mean(critic(real_ct)) + torch.mean(critic(fake_ct)) + args.lambda_gp * compute_gradient_penalty(critic, real_ct, fake_ct, device)
            loss_C.backward()
            optimizer_C.step()
            
            # Train Generator
            if i % args.n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, args.latent_dim).to(device)
                loss_G = -torch.mean(critic(generator(z)))
                loss_G.backward()
                optimizer_G.step()
                
        print(f"Epoch [{epoch:03d}/{args.epochs}] | Critic Loss: {loss_C.item():.4f} | Gen Loss: {loss_G.item():.4f}")
        
        # Checkpointing
        if epoch % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(args.out_dir, f"baseline_gen_epoch_{epoch:03d}.pth"))
            print("--> [CHECKPOINT SAVED]")

if __name__ == "__main__":
    main()