import torch
from model import SE3ScoreModel
from sde import VESDE, VPSDE
from ds_utils import get_dataloaders
from train import train
from sampler import pc_sampler_batch
from visual_util import plot_ca_backbone
import os

def main():
    # === Configs ===
    num_epochs = 30
    learning_rate = 1e-4
    checkpoint_path = "checkpoints/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Dataset & Dataloader ===
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # === Model, SDE, Optimizer ===
    model = SE3ScoreModel().to(device)
    # sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    sde = VPSDE(beta_0=0.1, beta_1=20.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Make folder if needed ===
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # === Train ===
    train(model, train_loader, optimizer, sde, num_epochs=num_epochs, save_path=checkpoint_path, device=device)

    # === Sampling ===
    model = SE3ScoreModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    lengths = [260, 280]  # Sample 2 domains of different lengths
    coords, batch = pc_sampler_batch(model, sde, lengths=lengths, device=device)
    for i in range(len(lengths)) :
        domain = coords[batch == i].cpu().numpy()
        plot_ca_backbone(domain)

if __name__ == "__main__":
    main()