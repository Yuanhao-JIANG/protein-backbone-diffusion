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
