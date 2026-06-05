"""
DDPM (Denoising Diffusion Probabilistic Models) - Paper Implementation
Based on: Ho et al., 2020 (https://arxiv.org/abs/2006.11239)

Key concepts:
- Forward process: gradually add noise to images over T timesteps
- Reverse process: learn to denoise at each step
- Training: predict noise at random timesteps
- Sampling: reverse from pure noise to generate images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================================
# 1. DIFFUSION SCHEDULE (Paper: Section 3.1)
# ============================================================================
class DiffusionSchedule:
    """Noise schedule defines how much noise to add at each timestep."""
    
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_steps = num_steps
        self.device = device
        
        # Linear schedule for betas (noise variance at each step)
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        
        # Derived quantities (Paper: equations 4-7)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # cumulative product
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        
        # Variance at each timestep (for sampling)
        self.variance = self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars)
    
    def add_noise(self, x0, t, noise):
        """
        Forward process: q(x_t|x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise
        Paper: Equation 4
        """
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise


# ============================================================================
# 2. U-NET ARCHITECTURE (Paper: Section 4)
# ============================================================================
class TimeEmbedding(nn.Module):
    """Embed timestep as sinusoidal embeddings (like positional encoding in Transformers)."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, t):
        # Sinusoidal embedding
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        emb = self.linear1(emb)
        emb = F.gelu(emb)
        emb = self.linear2(emb)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_channels),
        )
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(F.gelu(x))
        h = h + self.time_mlp(time_emb).view(h.shape[0], -1, 1, 1)
        h = F.gelu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """Simple U-Net for DDPM (Paper: Section 4)."""
    
    def __init__(self, channels=1, time_dim=128):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_dim)
        
        # Encoder (downsampling)
        self.enc1 = ResBlock(channels, 64, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResBlock(64, 128, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock(128, 256, time_dim)
        
        # Decoder (upsampling)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = ResBlock(256 + 128, 128, time_dim)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = ResBlock(128 + 64, 64, time_dim)
        
        # Output
        self.out = nn.Conv2d(64, channels, kernel_size=1)
    
    def forward(self, x, t):
        # Get time embedding
        time_emb = self.time_embed(t)
        
        # Encoder
        e1 = self.enc1(x, time_emb)
        e1_down = self.pool1(e1)
        
        e2 = self.enc2(e1_down, time_emb)
        e2_down = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(e2_down, time_emb)
        
        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, time_emb)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, time_emb)
        
        return self.out(d1)


# ============================================================================
# 3. DATA LOADING
# ============================================================================
def get_data_loader(batch_size=128, img_size=28):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ============================================================================
# 4. TRAINING (Paper: Algorithm 1)
# ============================================================================
def train(num_epochs=10, batch_size=128, num_steps=1000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop following DDPM Paper Algorithm 1:
    1. Sample x_0 from data
    2. Sample random timestep t
    3. Sample noise epsilon
    4. Compute noisy image x_t
    5. Predict noise and compute loss
    """
    
    # Setup
    schedule = DiffusionSchedule(num_steps=num_steps, device=device)
    model = UNet(channels=1, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = get_data_loader(batch_size=batch_size)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (x0, _) in enumerate(pbar):
            x0 = x0.to(device)
            batch_size_actual = x0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, num_steps, (batch_size_actual,), device=device)
            
            # Sample noise
            noise = torch.randn_like(x0)
            
            # Forward process: add noise to x0
            xt = schedule.add_noise(x0, t, noise)
            
            # Predict noise (model learns to denoise)
            pred_noise = model(xt, t)
            
            # Loss: MSE between predicted noise and actual noise
            loss = F.mse_loss(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / len(loader):.6f}")
    
    return model, schedule


# ============================================================================
# 5. SAMPLING (Paper: Algorithm 2)
# ============================================================================
@torch.no_grad()
def sample(model, schedule, num_samples=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Sampling loop following DDPM Paper Algorithm 2:
    1. Start from x_T ~ N(0, I)
    2. For t from T to 1:
       - Predict noise at current timestep
       - Compute mean of reverse process
       - Add noise (if t > 1)
    """
    
    model.eval()
    
    # Start from pure noise
    x = torch.randn(num_samples, 1, 28, 28, device=device)
    
    # Reverse diffusion process
    for t in tqdm(reversed(range(schedule.num_steps)), total=schedule.num_steps, desc="Sampling"):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Predict noise
        pred_noise = model(x, t_tensor)
        
        # Compute mean (Paper: Equation 11)
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]
        
        mean = (x - beta / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        
        # Add noise (except for last step)
        if t > 0:
            noise = torch.randn_like(x)
            variance = schedule.variance[t]
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean
    
    return x


# ============================================================================
# 6. VISUALIZATION
# ============================================================================
def plot_samples(samples, title="Generated Samples"):
    """Visualize generated images."""
    samples = samples.cpu().numpy()
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = samples[i, 0]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train
    print("\n=== Training DDPM ===")
    model, schedule = train(num_epochs=5, batch_size=128, num_steps=1000, device=device)
    
    # Sample
    print("\n=== Generating Samples ===")
    samples = sample(model, schedule, num_samples=16, device=device)
    plot_samples(samples, title="DDPM Generated MNIST Digits")
    
    # Save model
    torch.save(model.state_dict(), 'ddpm_model.pth')
    print("Model saved as ddpm_model.pth")
