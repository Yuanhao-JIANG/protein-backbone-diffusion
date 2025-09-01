# test.py
# some helper functions for document writing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_backbone_length_distribution(csv_file, bins=60, figsize=(4,3), save_path=None):
    """
    Plot the distribution of backbone lengths (L) from a CSV file.
    Assumes the file has a column "L".
    """
    df = pd.read_csv(csv_file)

    plt.figure(figsize=figsize)
    sns.histplot(df["L"], bins=bins, kde=True, stat="density", alpha=0.6, color="steelblue")

    plt.xlabel("Backbone length (L)")
    plt.ylabel("Density")
    plt.title("Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


plot_backbone_length_distribution('./benchmark/results_unet.csv', save_path="./figs/backbone_length_dist.png")
