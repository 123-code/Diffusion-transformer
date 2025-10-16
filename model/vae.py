import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model Architecture ---
# Encoder and Decoder with deeper layers, BatchNorm for stability.
# Decoder outputs log-variance for Gaussian NLL (better than fixed MSE).

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 20, input_height: int = 32, input_width: int = 32):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, input_height, input_width)
            x = self.convs(dummy_input)
            self.final_channels, self.final_height, self.final_width = x.shape[1:]
            flattened_size = x.numel()

        self.flatten = nn.Flatten()
        self.mu = nn.Linear(flattened_size, latent_dim)
        self.logvar = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

# Reparameterization trick
def reparameterize(mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(mu, device=mu.device)
    return mu + sigma * epsilon

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 20, final_channels: int = 256, final_height: int = 2, final_width: int = 2):
        super().__init__()
        self.final_channels = final_channels
        self.final_height = final_height
        self.final_width = final_width
        self.flattened_size = final_channels * final_height * final_width

        self.l1 = nn.Linear(latent_dim, self.flattened_size)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 6, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z):
        x = self.l1(z)
        x = x.view(x.shape[0], self.final_channels, self.final_height, self.final_width)
        x = self.deconvs(x)
        recons_mu = torch.tanh(x[:, :3, :, :])
        recons_log_var = F.softplus(x[:, 3:, :, :])  # Ensure positive variance
        return recons_mu, recons_log_var

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=20, input_height=32, input_width=32):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, input_height, input_width)
        self.decoder = Decoder(latent_dim,
                               self.encoder.final_channels,
                               self.encoder.final_height,
                               self.encoder.final_width)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        reconstructed_mu, reconstructed_log_var = self.decoder(z)
        return reconstructed_mu, reconstructed_log_var, mu, logvar

def vae_loss(reconstructed_mu, reconstructed_log_var, original, mu, logvar, kl_weight: float = 1.0):
    # Reconstruction Loss (Gaussian NLL, mean reduction)
    recon_loss = F.gaussian_nll_loss(reconstructed_mu, original, reconstructed_log_var, reduction='mean', full=True)

    # KL Divergence (mean reduction)
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_weight * kl_divergence

    # For logging
    with torch.no_grad():
        num_pixels_per_sample = original[0].numel()
        latent_dim = mu.size(1)
        recon_per_pixel = recon_loss / num_pixels_per_sample
        kl_per_dim = kl_divergence / latent_dim

    return total_loss, recon_loss.detach(), kl_divergence.detach(), recon_per_pixel, kl_per_dim

def train(model, train_loader, optimizer, epochs, device, kl_max: float = 0.5, kl_warmup_epochs: int = 40):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_recon_px = 0
        total_kl_dim = 0
        # Linear KL warmup to avoid posterior collapse
        kl_weight = min(kl_max, (epoch + 1) / max(1, kl_warmup_epochs) * kl_max)
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed_mu, reconstructed_log_var, mu, logvar = model(data)
            loss, recon_l, kl_l, recon_px, kl_dim = vae_loss(reconstructed_mu, reconstructed_log_var, data, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            total_recon_px += recon_px.item()
            total_kl_dim += kl_dim.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "vae.pth")
        avg_recon_px = total_recon_px / len(train_loader)
        avg_kl_dim = total_kl_dim / len(train_loader)
        print(
            f"Epoch {epoch+1}/{epochs}, kl_w={kl_weight:.3f}, "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f} (px {avg_recon_px:.5f}), "
            f"KL: {avg_kl:.4f} (per-dim {avg_kl_dim:.5f})"
        )

def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_recon_px = 0
    total_kl_dim = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            reconstructed_mu, reconstructed_log_var, mu, logvar = model(data)
            loss, recon_l, kl_l, recon_px, kl_dim = vae_loss(reconstructed_mu, reconstructed_log_var, data, mu, logvar, kl_weight=1.0)
            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            total_recon_px += recon_px.item()
            total_kl_dim += kl_dim.item()

    avg_loss = total_loss / len(test_loader)
    avg_recon = total_recon / len(test_loader)
    avg_kl = total_kl / len(test_loader)
    avg_recon_px = total_recon_px / len(test_loader)
    avg_kl_dim = total_kl_dim / len(test_loader)
    print(
        f"Test Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f} (px {avg_recon_px:.5f}), "
        f"KL: {avg_kl:.4f} (per-dim {avg_kl_dim:.5f})"
    )
    return avg_loss

def generate_samples(model, num_samples=16, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, model.encoder.mu.out_features).to(device)
        generated_mu, _ = model.decoder(z)
        return generated_mu

def save_generated_images(generated_images, filename="generated_samples.png"):
    import matplotlib.pyplot as plt

    # Convert to numpy and denormalize from [-1, 1] to [0, 1]
    images = (generated_images.clamp(-1, 1).cpu().numpy() + 1) / 2
    images = images.transpose(0, 2, 3, 1)

    # Create subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i, img in enumerate(images):
        if i < 16:
            axes[i].imshow(img, interpolation='nearest')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Generated images saved as {filename}")


def get_dataloaders(transform, batch_size: int = 128, num_workers: int = 2, data_root: str = './data'):
    """Try CelebA first; if download fails, fall back to CIFAR10."""
    try:
        train_dataset = datasets.CelebA(root=data_root, split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(root=data_root, split='test', download=True, transform=transform)
        used_name = 'CelebA'
    except Exception as e:
        print(f"CelebA unavailable ({e.__class__.__name__}): falling back to CIFAR10")
        train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        used_name = 'CIFAR10'

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader, used_name

if __name__ == "__main__":
    # CelebA transforms for 64x64 images
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    train_loader, test_loader, dataset_name = get_dataloaders(transform, batch_size=128, num_workers=2, data_root='./data')

    model = VAE(in_channels=3, latent_dim=128, input_height=64, input_width=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    epochs = 100

    print(f"Starting training on {dataset_name}...")
    train(model, train_loader, optimizer, epochs, device)
    test(model, test_loader, device)

    print("\nGenerating samples...")
    generated_images = generate_samples(model, num_samples=16, device=device)
    save_generated_images(generated_images)