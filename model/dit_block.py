import torch
import torch.nn as nn
import math
from einops import rearrange

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

class DiT(nn.Module):
    def __init__(self, latent_dim=1024, dim=512, depth=6, heads=8, mlp_ratio=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.depth = depth
        self.num_patches = latent_dim // dim  # Assume flat latent; adjust if spatial
        self.in_proj = nn.Linear(latent_dim, dim * self.num_patches)  # To tokens
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.out_proj = nn.Linear(dim * self.num_patches, latent_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(1, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )  # Simple t embed

    def forward(self, noisy_latent, t):
        b = noisy_latent.shape[0]
        t_emb = self.t_embed(t.float().unsqueeze(1))  # [B, dim]
        x = rearrange(self.in_proj(noisy_latent), 'b (n d) -> b n d', n=self.num_patches)  # To tokens
        for block in self.blocks:
            x = block(x, t_emb)
        x = rearrange(x, 'b n d -> b (n d)')
        return self.out_proj(x)  # Predicted noise