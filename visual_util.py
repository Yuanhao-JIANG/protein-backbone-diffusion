import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_ca_backbone(coords, title="Protein Backbone", save=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Plot backbone line
    ax.plot(x, y, z, '-o', linewidth=2, markersize=4)

    # Aesthetic adjustments
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    if save:
        plt.savefig(f'./figs/{len(coords)}.png')
    plt.show()

def plot_perturbation(coords, sde, save=False):
    times = [0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.995]
    n_plots = len(times)
    n_cols = 6
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, t in enumerate(times):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        mean, std = sde.marginal_prob(coords, torch.tensor([t]))
        noise = torch.randn_like(coords)
        x_t = mean + std.view(-1, 1) * noise
        ax.plot(x_t[:, 0], x_t[:, 1], x_t[:, 2], '-o', linewidth=2, markersize=4)
        ax.set_title(f't = {t:.3f}, std = {std.item():.3f}', fontsize=12)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])  # Keep an aspect ratio

    plt.tight_layout()
    if save:
        plt.savefig(f'./figs/perturbation_{len(coords)}.png')
    plt.show()
