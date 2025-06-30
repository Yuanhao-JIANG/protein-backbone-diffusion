import torch


@torch.no_grad()
def pc_sampler_batch(model, sde, lengths, num_steps=1000, snr=0.01, n_corr_steps=2, eps=1e-3, device='cpu'):
    """
    Args:
        model: trained score model
        sde: VPSDE instance
        lengths: list of integers, each the number of residues in a domain
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
        t_i_batch = t_i.expand(len(lengths))

        # --- (4) Corrector step (Langevin)
        for _ in range(n_corr_steps):
            score = model(x, batch=batch, t=t_i_batch)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(score.reshape(total_nodes, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(total_nodes, -1), dim=-1).mean()
            step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * sde.beta(t_i)

            x = x + step_size * score + torch.sqrt(2 * step_size) * noise

        # --- (5) Predictor step (Euler–Maruyama)
        score = model(x, batch=batch, t=t_i_batch)
        beta_t = sde.beta(t_i).view(-1, 1)
        drift = -0.5 * beta_t * x - beta_t * score
        diffusion = torch.sqrt(beta_t)

        z = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(-dt) * z

    return x, batch