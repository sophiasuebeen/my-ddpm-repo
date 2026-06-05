import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. toy data: mixture of two Gaussians
def sample_data(n):
    x1 = torch.randn(n//2, 1) * 0.4 - 2
    x2 = torch.randn(n//2, 1) * 0.4 + 2
    return torch.cat([x1, x2], dim=0)

# 2. time embedding + noise predictor
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / T
        return self.net(torch.cat([x, t], dim=1))

# 3. diffusion schedule
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

model = DenoiseNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. training: teach model to predict noise
for step in range(5000):
    x0 = sample_data(128)
    t = torch.randint(0, T, (128,))

    noise = torch.randn_like(x0)
    a_bar = alpha_bars[t].unsqueeze(1)

    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    pred_noise = model(xt, t)
    loss = ((pred_noise - noise) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())

# 5. sampling: start from noise and denoise step by step
@torch.no_grad()
def sample(n):
    x = torch.randn(n, 1)

    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        pred_noise = model(x, tt)

        beta = betas[t]
        alpha = alphas[t]
        alpha_bar = alpha_bars[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - beta / torch.sqrt(1 - alpha_bar) * pred_noise
        )

        if t > 0:
            x += torch.sqrt(beta) * torch.randn_like(x)

    return x

generated = sample(1000).squeeze()
real = sample_data(1000).squeeze()

plt.hist(real.numpy(), bins=50, alpha=0.5, label="real")
plt.hist(generated.numpy(), bins=50, alpha=0.5, label="generated")
plt.legend()
plt.show()