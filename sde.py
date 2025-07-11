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

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    def diffusion(self, x, t):
        return torch.sqrt(self.beta(t))

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


class CosineVPSDE:
    def __init__(self, s=0.008):
        self.s = s

    def _alpha_bar(self, t):
        return torch.cos((t + self.s) / (1 + self.s) * torch.pi / 2) ** 2

    def marginal_prob(self, x_0, t):
        alpha_bar = self._alpha_bar(t)
        mean = x_0 * torch.sqrt(alpha_bar)
        std = torch.sqrt(1. - alpha_bar)
        return mean, std

    def beta(self, t):
        # Compute beta(t) = -d log alpha_bar(t) / dt
        s = self.s
        return torch.tan((t + s) / (1 + s) * torch.pi / 2) * torch.pi / (1 + s)

    def drift(self, x, t):
        beta_t = self.beta(t)
        return -0.5 * beta_t * x

    def diffusion(self, t):
        beta_t = self.beta(t)
        return torch.sqrt(beta_t)
