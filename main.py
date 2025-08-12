import os
import torch
import numpy as np
from model import SE3ScoreModel, UNetScoreModel, GATv2ScoreModel, GINEScoreModel, PNAScoreModel, GPSScoreModel, GraphTransformerScoreModel, GraphTransformerWOPosScoreModel, GraphUNetScoreModel
from sde import VESDE, VPSDE, CosineVPSDE
from ds_utils import get_dataloaders, CADataset
from train import train
import pandas as pd
from sampler import pc_sampler_batch
from benchmark import run_benchmark, load_all, summarise_benchmarks, plot_metric_distribution, plot_metric_box_violin, plot_metric_vs_length

def main():
    # === Configs ===
    num_epochs = 300
    batch_size = 64
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    truncate = False
    mode = 'summary'  # 'train', 'sample', 'benchmark', 'summary'
    print("Using device:", device)

    # === Dataset & Dataloader ===
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, truncate=truncate)

    # === Model ===
    # model = SE3ScoreModel().to(device)
    # model = GATv2ScoreModel().to(device)
    # model = GINEScoreModel().to(device)
    # model = PNAScoreModel(train_loader).to(device)
    # model = GPSScoreModel().to(device)
    # model = GraphUNetScoreModel().to(device)
    # model = GraphTransformerWOPosScoreModel().to(device)
    model = GraphTransformerScoreModel().to(device)
    # model = UNetScoreModel(truncate=truncate).to(device)
    print(f'{model.__class__.__name__}. Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    # sde = VPSDE(beta_0=0.1, beta_1=20.0)
    sde = CosineVPSDE(s=0.001)

    # === Make folder if needed ===
    checkpoint_path = f"checkpoints/{model.__class__.__name__}.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if mode == 'train':
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        train(model, train_loader, optimizer, sde, num_epochs=num_epochs, save_path=checkpoint_path, device=device)
    elif mode == 'sample':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        lengths = [189, 200]  # Sample 2 domains of different lengths
        coords, batch = pc_sampler_batch(model, sde, lengths=lengths, n_corr_steps=3, plot=True, device=device)
    elif mode == 'benchmark':
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        run_benchmark(
            model=model,
            sde=sde,
            npy_dir="dataset/ca_coords",
            out_csv="benchmark/results.csv",
            device=device,
            limit=None,
            max_nodes_per_batch=15000,
            save_gen_pdb_dir=None
        )
    elif mode == 'summary':
        files = {
            "TransformerConv+PE (5-layer Res)": "./benchmark/results_gnn_pos.csv",
            "UNet (padded)": "./benchmark/results_unet.csv",
            # "TransformerConv w/o PE": "./benchmark/results_gnn_wo_pos.csv",
        }
        df_all = load_all(files)

        summary = summarise_benchmarks(df_all, save_csv="./benchmark/benchmark_summary.csv")
        pd.set_option('display.max_columns', None)
        print(summary)

        # 1) Distribution
        plot_metric_distribution(df_all, metric="rmsd", add_ref_lines=False, save_path="./benchmark/rmsd_dist.png")
        plot_metric_distribution(df_all, metric="tm", add_ref_lines=False, save_path="./benchmark/tm_dist.png")

        # 2) Box / Violin
        plot_metric_box_violin(df_all, metric="rmsd", kind="box", add_ref_lines=False, save_path="./benchmark/rmsd_box.png")
        plot_metric_box_violin(df_all, metric="tm", kind="box", add_ref_lines=False, save_path="./benchmark/tm_box.png")

        # 3) Scatter vs. L with trend lines
        plot_metric_vs_length(df_all, metric="rmsd", add_linear_trend=True, add_ref_lines=False, save_path="./benchmark/rmsd_vs_L.png")
        plot_metric_vs_length(df_all, metric="tm", add_linear_trend=True, add_ref_lines=False, save_path="./benchmark/tm_vs_L.png")
    else:
        raise ValueError(f"Unknown mode: {mode}")



if __name__ == "__main__":
    main()