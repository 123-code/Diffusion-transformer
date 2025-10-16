import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.vae import VAE  # Your VAE file
from diffusion import DiffusionProcess, LinearSchedule
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data (CIFAR-10 latents from VAE)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dl = DataLoader(ds, batch_size=32, shuffle=True)

# Load VAE, extract latents
vae = VAE(in_channels=3, latent_dim=128, input_height=64, input_width=64).to(device)
vae.load_state_dict(torch.load('vae.pth', map_location=device))
vae.eval()

def get_latents(batch):
    with torch.no_grad():
        _, _, z, _ = vae(batch)
        print(f"Latent shape: {z.shape}")  # Debug: [B, 128]
        return z  # [B, 128]

# DiT block components
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = math.sqrt(dim // heads)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) + x  # Residual

class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x))) + x  # Residual

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.t_proj = nn.Linear(dim, dim * 2)  # Scale + shift

    def forward(self, x, t_emb):
        normed = self.norm(x)
        scale, shift = self.t_proj(t_emb).chunk(2, dim=-1)
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)  # Broadcast to seq
        return normed * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = SelfAttention(dim, heads)
        self.ffn = FFN(dim, mlp_ratio)
        self.adaln1 = AdaLN(dim)
        self.adaln2 = AdaLN(dim)

    def forward(self, x, t_emb):
        x1 = self.adaln1(x, t_emb)
        x1 = self.attn(x1) + x  # Residual
        x2 = self.adaln2(x1, t_emb)
        x2 = self.ffn(x2) + x1  # Residual
        return x2

# Updated DiT for flat latents (single token, no patching)
class DiT(nn.Module):
    def __init__(self, latent_dim=128, dim=512, depth=6, heads=8, mlp_ratio=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.depth = depth
        self.in_proj = nn.Linear(latent_dim, dim)  # Flat to dim (single token)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.out_proj = nn.Linear(dim, latent_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, noisy_latent, t):
        b = noisy_latent.shape[0]
        t_emb = self.t_embed(t.float().unsqueeze(1))  # [B, dim]
        x = self.in_proj(noisy_latent).unsqueeze(1)  # [B, 1, dim] (single token)
        for block in self.blocks:
            x = block(x, t_emb)
        x = x.squeeze(1)  # Back to [B, dim]
        return self.out_proj(x)  # Predicted noise [B, latent_dim]

# Custom diffusion for 2D latents
class LatentDiffusionProcess:
    def __init__(self, timesteps=1000):
        self.schedule = LinearSchedule(timesteps)
        self.timesteps = timesteps

    def training_loss(self, model, latents, t):
        noise = torch.randn_like(latents)
        t_idx = torch.randint(0, self.timesteps, (latents.shape[0],)).long().to(latents.device)
        # Add dummy dimensions for compatibility with LinearSchedule
        sqrt_alphas_cumprod = torch.sqrt(self.schedule.alphas_cumprod[t_idx]).view(-1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.schedule.alphas_cumprod[t_idx]).view(-1, 1)
        noisy_latents = sqrt_alphas_cumprod * latents + sqrt_one_minus_alphas_cumprod * noise
        pred_noise = model(noisy_latents, t_idx)
        return F.mse_loss(pred_noise, noise)

# DiT and diffusion
dit = DiT(latent_dim=128, dim=512, depth=6).to(device)
opt = torch.optim.AdamW(dit.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
diffusion = LatentDiffusionProcess(timesteps=1000)

# Training
for epoch in range(20):
    total_loss = 0
    for imgs, _ in dl:
        imgs = imgs.to(device)
        latents = get_latents(imgs)
        t = torch.randint(0, diffusion.timesteps, (latents.shape[0],)).to(device)
        loss = diffusion.training_loss(dit, latents, t)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(dl)
    print(f"Epoch {epoch+1}/20, Avg Loss: {avg_loss:.4f}")
    if epoch % 5 == 0:
        torch.save(dit.state_dict(), f"dit_epoch_{epoch}.pt")

# Generation test
def generate_faces(dit, diffusion, vae, num_samples=16, steps=50):
    with torch.no_grad():
        z = torch.randn(num_samples, 128).to(device)  # Start with noise
        for i in range(steps, 0, -1):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = dit(z, t)
            # Denoising step using the schedule
            alpha_t = torch.sqrt(diffusion.schedule.alphas_cumprod[i-1]).to(device)
            alpha_t_prev = torch.sqrt(diffusion.schedule.alphas_cumprod[i-2]).to(device) if i > 1 else torch.tensor(0.0).to(device)
            beta_t = 1 - diffusion.schedule.alphas_cumprod[i-1]

            pred_x0 = (z - torch.sqrt(beta_t) * pred_noise) / torch.sqrt(diffusion.schedule.alphas_cumprod[i-1])
            z = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * torch.randn_like(z)

        recon_mu, _ = vae.decoder(z)
        return recon_mu

gens = generate_faces(dit, diffusion, vae)
# Save with your save_generated_images(gens)