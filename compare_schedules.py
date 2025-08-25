# compare_schedules.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sde import VPSDE, CosineVPSDE


def perturb_image(x, sde, t):
    mean, std = sde.marginal_prob(x, torch.tensor([t]))
    noise = torch.randn_like(x)
    x_t = mean + std.view(-1, 1, 1, 1) * noise
    return x_t.clamp(0, 1)


# Load image and normalise to [0,1]
img_path, img_size = "./docs/imgs/ox_logo.png", 96
img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
img_tensor = torch.from_numpy(np.array(img) / 255.0).float().permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

linear_sde = VPSDE(beta_0=0.1, beta_1=30.0)
cosine_sde = CosineVPSDE(s=0.001)

# Choose timesteps to visualise
timesteps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.55, 0.75, 1.0]

fig, axes = plt.subplots(2, len(timesteps), figsize=(2*len(timesteps), 4.5))
axes[0, 0].set_ylabel("Linear", fontsize=25, labelpad=10)
axes[1, 0].set_ylabel("Cosine", fontsize=25, labelpad=10)

for j, t in enumerate(timesteps):
    # Linear schedule
    noisy_lin = perturb_image(img_tensor, linear_sde, t)
    axes[0, j].imshow(noisy_lin.squeeze(0).permute(1, 2, 0).numpy())
    # axes[0, j].axis("off")

    # Cosine schedule
    noisy_cos = perturb_image(img_tensor, cosine_sde, t)
    axes[1, j].imshow(noisy_cos.squeeze(0).permute(1, 2, 0).numpy())
    # axes[1, j].axis("off")

    axes[1, j].set_xlabel(f"t = {t:.2f}", fontsize=25, labelpad=10)
    for ax in [axes[0, j], axes[1, j]]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

plt.tight_layout()
plt.savefig("./figs/linear_vs_cosine.png", dpi=200)
plt.show()
