import torch
from model import get_noise_conditioned_score
import matplotlib.pyplot as plt
from model import SE3ScoreModel


@torch.no_grad()
def  pc_sampler_batch(model, sde, lengths, num_steps=1000, snr=0.15, n_corr_steps=2, eps=1e-3, plot=False, device='cpu'):
    """
    Args:
        model: trained score model
        sde: VPSDE instance
        lengths: list of integers, each the number of residues in a domain
        num_steps: number of predictor steps per domain
        snr: signal-to-noise ratio for Langevin correction
        n_corr_steps: number of Langevin correction steps
        eps: starting time
        plot: whether to plot the denoised coordinates
        device: device to run on
    Returns:
        Tensor of shape [sum(lengths), 3] (coordinates)
        Tensor of shape [sum(lengths)] (batch indices)
    """
    model.eval()

    total_nodes = sum(lengths)

    # --- (1) Initialize Gaussian noise
    x = torch.randn(total_nodes, 3, device=device)

    # --- (2) Build batch vector
    batch = torch.cat([
        torch.full((l,), i, dtype=torch.long, device=device)
        for i, l in enumerate(lengths)
    ], dim=0)

    # --- (3) Time steps
    t_array = torch.linspace(1.0-eps, eps, num_steps, device=device)
    dt = torch.tensor(-1.0 / num_steps)

    if plot:
        n_cols = 5
        n_rows = 2
        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        j = 0

    for i in range(num_steps):
        t_i = t_array[i]
        t_i_batch = t_i.expand(len(lengths))

        # --- (4) Corrector step (Langevin)
        for _ in range(n_corr_steps):
            score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(total_nodes, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(total_nodes, -1), dim=-1).mean()
            if isinstance(model, SE3ScoreModel):
                step_size = torch.tensor(0.01)
            else:
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        # --- (5) Predictor step (Euler–Maruyama)
        score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
        drift = sde.drift(x, t_i) - sde.diffusion(x, t_i) ** 2 * score
        diffusion = sde.diffusion(x, t_i)

        z = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(-dt) * z

        print(f"Time {t_i:.3f}, step_size = {step_size:.6f}, grad_norm = {grad_norm:.4f}, x min: {x.abs().min().item():.3f}, x max: {x.abs().max().item():.3f}, mean: {x.norm(dim=-1).mean().item():.3f}")

        if plot and (i % 120 == 0 or i == num_steps - 1):
            x_t = x[batch == 0].cpu().numpy()
            ax = fig.add_subplot(n_rows, n_cols, j + 1, projection='3d')
            ax.plot(x_t[:, 0], x_t[:, 1], x_t[:, 2], '-o', linewidth=2, markersize=4)
            ax.set_title(f't = {t_i:.3f}', fontsize=12)
            ax.set_box_aspect([1, 1, 1])  # Keep an aspect ratio
            j+=1

    if plot:
        plt.tight_layout()
        plt.savefig(f'./figs/{model.__class__.__name__}_denoising_{len(x_t)}.png')
        plt.show()

    return x, batch

@torch.no_grad()
def  batch_sampler(model, sde, lengths, num_steps=1000, snr=0.12, n_corr_steps=3, eps=1e-3, device='cuda'):
    model.eval()

    total_nodes = sum(lengths)

    # --- (1) Initialize Gaussian noise
    x = torch.randn(total_nodes, 3, device=device)

    # --- (2) Build batch vector
    batch = torch.cat([
        torch.full((l,), i, dtype=torch.long, device=device)
        for i, l in enumerate(lengths)
    ], dim=0)

    # --- (3) Time steps
    t_array = torch.linspace(1.0-eps, eps, num_steps, device=device)
    dt = torch.tensor(-1.0 / num_steps)

    for i in range(num_steps):
        t_i = t_array[i]
        t_i_batch = t_i.expand(len(lengths))

        # --- (4) Corrector step (Langevin)
        for _ in range(n_corr_steps):
            score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(total_nodes, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(total_nodes, -1), dim=-1).mean()
            if isinstance(model, SE3ScoreModel):
                step_size = torch.tensor(0.01)
            else:
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        # --- (5) Predictor step (Euler–Maruyama)
        score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
        drift = sde.drift(x, t_i) - sde.diffusion(x, t_i) ** 2 * score
        diffusion = sde.diffusion(x, t_i)

        z = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(-dt) * z

    return x, batch
