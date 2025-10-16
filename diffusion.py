import torch
import math

class LinearSchedule:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_betas(self):
        return self.betas

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        pred_noise = model(x, t)
        x_start = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
        dir_xt = torch.randn_like(x)
        x_tilde = sqrt_one_minus_alphas_cumprod_t * x_start + sqrt_alphas_cumprod_t * dir_xt
        return x_tilde

class DiffusionProcess:
    def __init__(self, timesteps=1000):
        self.schedule = LinearSchedule(timesteps)
        self.timesteps = timesteps

    def training_loss(self, model, latents, t):
        noise = torch.randn_like(latents)
        t_idx = torch.randint(0, self.timesteps, (latents.shape[0],)).long().to(latents.device)
        noisy_latents = self.schedule.q_sample(latents, t_idx, noise)
        pred_noise = model(noisy_latents, t_idx)
        return F.mse_loss(pred_noise, noise)

    def sample(self, model, shape, steps=50):
        x = torch.randn(shape).to(model.device)
        for i in range(steps, 0, -1):
            t = torch.full((shape[0],), i, device=model.device, dtype=torch.long)
            x = self.schedule.p_sample(model, x, t)
        return x