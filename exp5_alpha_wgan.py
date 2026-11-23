import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# Generator / Image Critic
# Same generator and critic as exp1_train_gan.py.
# ==========================================
class Generator3D(nn.Module):
    """Maps a latent vector to an output_size^3 CT patch. Starts from a 4^3 volume
    and adds one upsampling stage per doubling of output_size."""
    def __init__(self, latent_dim=128, features=64, output_size=32):
        super().__init__()
        self.init_size = 4
        num_stages = round(np.log2(output_size / self.init_size))
        assert self.init_size * (2 ** num_stages) == output_size, \
            f"output_size={output_size} must be init_size({self.init_size}) * a power of 2"

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
                layers.append(nn.Sigmoid())
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

# ==========================================
# Encoder: deterministic (no KL term); the code critic enforces the prior.
# Uses the same fixed-bottleneck layout as VAE3D in phase3_train_VAE.py.
# ==========================================
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

class Encoder3D(nn.Module):
    """Deterministic encoder: real CT patch -> single latent code z_hat.
    Same fixed-bottleneck layout as VAE3D."""
    def __init__(self, latent_dim=128, input_size=32, bottleneck_spatial=4, max_channels=256):
        super().__init__()
        num_stages = round(np.log2(input_size / bottleneck_spatial))
        assert bottleneck_spatial * (2 ** num_stages) == input_size, \
            f"input_size={input_size} must be bottleneck_spatial({bottleneck_spatial}) * a power of 2"

        channel_list = [32]
        for _ in range(num_stages):
            channel_list.append(min(channel_list[-1] * 2, max_channels))

        layers = [
            nn.Conv3d(1, channel_list[0], kernel_size=3, stride=1, padding=1),
            ResBlock3D(channel_list[0]),
        ]
        for i in range(1, len(channel_list)):
            layers.append(nn.Conv3d(channel_list[i - 1], channel_list[i], kernel_size=3, stride=2, padding=1))
            layers.append(ResBlock3D(channel_list[i]))
        self.encoder = nn.Sequential(*layers)

        bottleneck_flat = channel_list[-1] * (bottleneck_spatial ** 3)
        self.fc = nn.Linear(bottleneck_flat, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

# ==========================================
# Code critic, WGAN-GP style, discriminates z~N(0,I) (real) vs E(x) (fake)
# ==========================================
class CodeCritic(nn.Module):
    def __init__(self, latent_dim=128, hidden=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)
        )

    def forward(self, z):
        return self.model(z)

# ==========================================
# Dataset (same as exp1_train_gan.py)
# ==========================================
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

# ==========================================
# Generalized WGAN-GP gradient penalty (works for 5D image tensors and 2D code tensors)
# ==========================================
def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha_shape = [real_samples.size(0)] + [1] * (real_samples.dim() - 1)
    alpha = torch.rand(*alpha_shape).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = Variable(torch.ones(real_samples.size(0), 1).to(device), requires_grad=False)

    gradients = grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# ==========================================
# Training Loop
# ==========================================
def setup_args():
    parser = argparse.ArgumentParser(description="Exp 5: Train 3D alpha-WGAN baseline")
    parser.add_argument("--data-dir", type=str, default="./output/phase1_data")
    parser.add_argument("--out-dir", type=str, default="./output/models/alpha_wgan")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--lambda-gp", type=float, default=10.0)
    parser.add_argument("--lambda-recon", type=float, default=100.0)
    parser.add_argument("--patch-size", type=int, default=32)
    return parser.parse_args()

def main():
    args = setup_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing alpha-WGAN Baseline on: {device}")

    encoder = Encoder3D(latent_dim=args.latent_dim, input_size=args.patch_size).to(device)
    generator = Generator3D(latent_dim=args.latent_dim, output_size=args.patch_size).to(device)
    critic = Critic3D(input_size=args.patch_size).to(device)
    code_critic = CodeCritic(latent_dim=args.latent_dim).to(device)

    optimizer_GE = optim.Adam(list(generator.parameters()) + list(encoder.parameters()), lr=args.lr, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optimizer_C = optim.Adam(code_critic.parameters(), lr=args.lr, betas=(0.0, 0.9))

    criterion_L1 = nn.L1Loss()

    dataloader = DataLoader(RealCTDataset(args.data_dir), batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"Starting alpha-WGAN Baseline Training for {args.epochs} Epochs...")

    history_D, history_C, history_GE = [], [], []

    global_loop = tqdm(range(1, args.epochs + 1), desc="Overall Progress", unit="epoch")
    for epoch in global_loop:
        epoch_d, epoch_c, epoch_ge = 0.0, 0.0, 0.0
        n_batches, n_ge_updates = 0, 0

        batch_loop = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{args.epochs}]", leave=False)
        for i, real_ct in enumerate(batch_loop):
            real_ct = real_ct.to(device)
            batch_size = real_ct.size(0)

            z_prior = torch.randn(batch_size, args.latent_dim).to(device)
            z_hat = encoder(real_ct)
            fake_prior = generator(z_prior).detach()
            fake_recon = generator(z_hat).detach()
            z_hat_detached = z_hat.detach()

            # Train Image Critic (real vs {prior-generated, reconstructed})
            optimizer_D.zero_grad()
            gp_prior = compute_gradient_penalty(critic, real_ct, fake_prior, device)
            gp_recon = compute_gradient_penalty(critic, real_ct, fake_recon, device)
            loss_D = (-torch.mean(critic(real_ct))
                      + 0.5 * (torch.mean(critic(fake_prior)) + torch.mean(critic(fake_recon)))
                      + args.lambda_gp * 0.5 * (gp_prior + gp_recon))
            loss_D.backward()
            optimizer_D.step()

            # Train Code Critic (z ~ N(0,I) vs E(x))
            optimizer_C.zero_grad()
            gp_code = compute_gradient_penalty(code_critic, z_prior, z_hat_detached, device)
            loss_C = -torch.mean(code_critic(z_prior)) + torch.mean(code_critic(z_hat_detached)) + args.lambda_gp * gp_code
            loss_C.backward()
            optimizer_C.step()

            epoch_d += loss_D.item()
            epoch_c += loss_C.item()
            n_batches += 1

            # Train Generator + Encoder jointly every n_critic steps
            if i % args.n_critic == 0:
                optimizer_GE.zero_grad()
                z_prior2 = torch.randn(batch_size, args.latent_dim).to(device)
                z_hat2 = encoder(real_ct)
                fake_prior2 = generator(z_prior2)
                fake_recon2 = generator(z_hat2)

                loss_adv_img = -torch.mean(critic(fake_prior2)) - torch.mean(critic(fake_recon2))
                loss_recon = criterion_L1(fake_recon2, real_ct)
                loss_adv_code = -torch.mean(code_critic(z_hat2))

                loss_GE = loss_adv_img + args.lambda_recon * loss_recon + loss_adv_code
                loss_GE.backward()
                optimizer_GE.step()

                epoch_ge += loss_GE.item()
                n_ge_updates += 1

            batch_loop.set_postfix(D=loss_D.item(), C=loss_C.item())

        avg_d = epoch_d / n_batches
        avg_c = epoch_c / n_batches
        avg_ge = epoch_ge / max(1, n_ge_updates)
        global_loop.write(f"Epoch [{epoch:03d}/{args.epochs}] | D Loss: {avg_d:.4f} | Code-C Loss: {avg_c:.4f} | G+E Loss: {avg_ge:.4f}")

        history_D.append(avg_d)
        history_C.append(avg_c)
        history_GE.append(avg_ge)

        if epoch % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(args.out_dir, f"alpha_wgan_gen_epoch_{epoch:03d}.pth"))
            torch.save(encoder.state_dict(), os.path.join(args.out_dir, f"alpha_wgan_enc_epoch_{epoch:03d}.pth"))
            global_loop.write("--> [CHECKPOINT SAVED]")

    print("Generating alpha-WGAN performance plot...")
    epochs_range = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.plot(epochs_range, history_D, label='Image Critic Loss', color='red', linewidth=2)
    ax1.plot(epochs_range, history_C, label='Code Critic Loss', color='purple', linewidth=2)
    ax1.plot(epochs_range, history_GE, label='Generator+Encoder Loss', color='blue', linewidth=2)
    ax1.legend(loc='upper right')
    plt.title('3D alpha-WGAN Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(args.out_dir, "alpha_wgan_training_metrics.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
