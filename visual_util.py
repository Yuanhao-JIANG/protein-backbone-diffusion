import numpy as np
import matplotlib.pyplot as plt


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
