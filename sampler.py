import torch
from model import get_noise_conditioned_score


@torch.no_grad()
def  pc_sampler_batch(model, sde, lengths, num_steps=1000, snr=0.01, n_corr_steps=2, eps=1e-3, device='cpu'):
    """
    Args:
        model: trained score model
        sde: VPSDE instance
        lengths: list of integers, each the number of residues in a domain
        num_steps: number of predictor steps per domain
        snr: signal-to-noise ratio for Langevin correction
        n_corr_steps: number of Langevin correction steps
        eps: starting time
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
    t_array = torch.linspace(1.0, eps, num_steps, device=device)
    dt = torch.tensor(-1.0 / num_steps)

    for i in range(num_steps):
        t_i = t_array[i]
        print(f't = {t_i}')
        t_i_batch = t_i.expand(len(lengths))

        # --- (4) Corrector step (Langevin)
        for _ in range(n_corr_steps):
            score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(total_nodes, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(total_nodes, -1), dim=-1).mean()
            step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * sde.beta(t_i)
            print(f'step_size = {step_size}, grad_norm = {grad_norm}')
            # step_size = torch.tensor(0.01)

            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        # --- (5) Predictor step (Euler–Maruyama)
        score = get_noise_conditioned_score(model, x, batch, t_i_batch, sde)
        drift = sde.drift(x, t_i) - sde.diffusion(x, t_i) ** 2 * score
        diffusion = sde.diffusion(x, t_i)

        z = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(-dt) * z

        print(f"Step {i}, x min: {x.abs().min().item():.2f}, x max: {x.abs().max().item():.2f}, mean: {x.norm(dim=-1).mean().item():.2f}")

    return x, batch