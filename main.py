import torch
from model import SE3ScoreModel, UNetScoreModel, GATv2ScoreModel, GINEScoreModel, PNAScoreModel, TransformerScoreModel, GPSScoreModel
from sde import VESDE, VPSDE, CosineVPSDE
from ds_utils import get_dataloaders
from train import train
from sampler import pc_sampler_batch
import os

def main():
    # === Configs ===
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Dataset & Dataloader ===
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, truncate=True)

    # === Model, SDE, Optimizer ===
    # model = SE3ScoreModel().to(device)
    # model = UNetScoreModel().to(device)
    # model = GATv2ScoreModel().to(device)
    # model = GINEScoreModel().to(device)
    # model = PNAScoreModel(train_loader).to(device)
    model = TransformerScoreModel().to(device)
    # model = GPSScoreModel().to(device)
    print(f'{model.__class__.__name__}. Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    # sde = VPSDE(beta_0=0.1, beta_1=20.0)
    sde = CosineVPSDE(s=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # === Make folder if needed ===
    checkpoint_path = f"checkpoints/{model.__class__.__name__}.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # === Train ===
    train(model, train_loader, optimizer, sde, num_epochs=num_epochs, save_path=checkpoint_path, device=device)

    # === Sampling ===
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    lengths = [20, 20]  # Sample 2 domains of different lengths
    coords, batch = pc_sampler_batch(model, sde, lengths=lengths, n_corr_steps=1, plot=True, device=device)

if __name__ == "__main__":
    main()