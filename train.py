import os
import torch
from tqdm import tqdm
from model import get_noise_conditioned_score

def loss_fn(model, coords, batch, sde, eps=1e-5):
    B = int(batch.max().item()) + 1

    # Sample one timestep per structure
    t = torch.rand(B, device=coords.device) * (1.0 - eps) + eps  # [B]

    # Expand t to each node
    t_nodes = t[batch]                                           # [total_nodes]

    # Compute marginal mean and std from diffusion SDE
    mean, std = sde.marginal_prob(coords, t_nodes)               # [total_nodes,3], [total_nodes]

    # Add noise to the coordinates
    noise = torch.randn_like(coords)
    x_t = mean + std * noise                                     # [total_nodes,3]

    # Forward pass through the score model
    score = get_noise_conditioned_score(model, x_t, batch, t, sde)  # [total_nodes,3]

    # Compute the denoising score-matching loss
    residual = score * std + noise                               # [total_nodes,3]

    # Node-level squared loss
    loss_per_node = torch.sum(residual**2, dim=-1)               # [total_nodes]

    # Graph-level accumulation
    loss_per_graph = torch.zeros(B, device=coords.device).index_add_(0, batch, loss_per_node)  # [B]

    # Average loss over batch
    return loss_per_graph.mean()

def train(model, dataloader, optimizer, sde, num_epochs=50, save_path='checkpoints/model.pt', device='cuda'):
    device = torch.device(device)
    model.to(device)
    model.train()

    for epoch in range(1, num_epochs+1):
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            coords_list = batch if isinstance(batch, (list, tuple)) else [batch]

            # from visual_util import plot_perturbation
            # coords = coords_list[0]
            # plot_perturbation(coords, sde)
            # exit()

            coords = torch.cat(coords_list, dim=0).to(device)  # Flattened coords [total_nodes,3]

            batch_idx = torch.cat([
                torch.full((c.shape[0],), i, dtype=torch.long, device=device)
                for i, c in enumerate(coords_list)
            ], dim=0)  # Batch indices [total_nodes]

            optimizer.zero_grad()
            loss = loss_fn(model, coords, batch_idx, sde)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

        # Save model checkpoint
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
