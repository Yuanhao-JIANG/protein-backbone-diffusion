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
