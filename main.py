import torch
from model import SE3ScoreModel
from sde import VESDE
from ds_utils import get_dataloaders
from train import train
import os

def main():
    # === Configs ===
    num_epochs = 5
    learning_rate = 1e-3
    checkpoint_path = "checkpoints/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset & Dataloader ===
    train_loader, val_loader, test_loader = get_dataloaders()

    # === Model, SDE, Optimizer ===
    model = SE3ScoreModel().to(device)
    sde = VESDE(sigma_min=0.01, sigma_max=50.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Make folder if needed ===
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # === Train ===
    train(model, train_loader, optimizer, sde, num_epochs=num_epochs, save_path=checkpoint_path, device=device)


if __name__ == "__main__":
    main()