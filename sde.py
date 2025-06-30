import torch

class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t):
        """Noise scale at time t in [0, 1]"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(self, x_0, t):
        """Mean and Standard deviation of p(x_t | x_0)"""
        return x_0, self.sigma(t)

    # def perturb(self, x0, t):
    #     """
    #     Sample x_t ~ p(x_t | x_0)
    #     Returns: x_t, noise epsilon
    #     """
    #     eps = torch.randn_like(x0)
    #     std = self.marginal_prob_std(t).view(-1, 1, 1)  # broadcast for batch
    #     x_t = x0 + eps * std
    #     return x_t, eps

    # def score_from_eps(self, eps, t):
    #     """Analytic score: nabla log p(x_t | x_0) = -epsilon / sigma(t)"""
    #     std = self.marginal_prob_std(t).view(-1, 1, 1)
    #     return -eps / std

    # def dsm_weight(self, t):
    #     """Weight lambda(t) in DSM loss (typically sigma(t)^2 for VE SDE)"""
    #     std = self.marginal_prob_std(t)
    #     return std ** 2


class VPSDE:
    def __init__(self, beta_0=0.1, beta_1=20.0):
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def beta(self, t):
        """Beta schedule: linear interpolation between beta_0 and beta_1"""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def marginal_log_mean_coeff(self, t):
        """Compute log of mu(t)"""
        # \int beta(s) ds = beta_0 * t + 0.5 * (beta_1 - beta_0) * t^2
        return -0.5 * (self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t**2)

    def marginal_prob(self, x_0, t):
        """
        Returns:
            mean: mu(t) * x_0
            std: sqrt(1 - mu(t)^2)
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)[:, None]
        mean = x_0 * torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        """Diffusion coefficient (std of the noise term in forward SDE)"""
        return torch.sqrt(self.beta(t))
