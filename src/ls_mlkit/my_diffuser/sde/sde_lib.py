from .base_sde import SDE
import torch
from torch import Tensor
import numpy as np
from typing import Tuple
from overrides import override


class VPSDE(SDE):
    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20, n_discretization_steps: int = 1000, ndim_micro_shape: int = 2
    ):
        """Construct a Variance Preserving SDE.

        Args:
        ------
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          n_discretization_steps: number of discretization steps
          ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(
            beta_min / n_discretization_steps, beta_max / n_discretization_steps, n_discretization_steps
        )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # mean
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # std

    @property
    @override
    def T(self) -> float:
        return 1

    @override
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""
        continuous DDPM SDE
        $$
        \begin{align*}
        dx &= -\frac{1}{2}\beta_t x dt + \sqrt{\beta_t} dw
        \end{align*}
        $$
        Args:
            x:
            t: (macro_shape)
            mask:
        Returns:
            drift: shape = x.shape
            diffusion: shape=x.macro_shape
        """
        macro_shape = x.shape[: self.ndim_micro_shape]
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)]) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    @override
    def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        """DDPM discretization."""
        timestep = (t * (self.n_discretization_steps - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha).view(alpha.shape[0], *[1 for _ in range(self.ndim_of_a_sample)]) * x - x
        g = sqrt_beta
        return f, g

    def get_target_score(self, x_0: Tensor, x_t: Tensor, t: Tensor, mask: Tensor, continuous: bool = False) -> Tensor:
        r"""
        $$
        p_{0t} (x_t|x_0) =
        \nabla_{x_t} \ln p_{0t} (x_t|x_0)
        $$
        """
        mu = None  # $$E_{x_t\sim p_{0t}(x_t|x_0)}[x_t]$$

        sigma = None  # $$\sqrt{Var_{x_t\sim p_{0t}(x_t|x_0)}[x_t]}$$

        macro_shape = x_t.shape[: self.ndim_micro_shape]

        if continuous:
            mu, sigma = self.marginal_prob(x_0, t, mask=None)
        else:
            mu = self.sqrt_alphas_cumprod[t].view(*macro_shape, *[1 for _ in range(self.ndim_of_a_sample)]) * x_0
            sigma = self.sqrt_1m_alphas_cumprod[t].view(*macro_shape, *[1 for _ in range(self.ndim_of_a_sample)])
        score = -(x_t - mu) / sigma**2
        return score

    def marginal_prob(self, x_0: Tensor, t: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
		$$
		p_{0t} (x_t|x_0)
		$$
		$$
		\begin{align*}
			\gamma = -\frac{1}{4}t^2 (\beta_1 - \beta_0) - \frac{1}{2} t  \beta_0\\
			mean = e^{\gamma} * x \\
			std = \sqrt{1 - e^{2 \gamma }}
		\end{align*}
		$$
		"""
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        macro_shape = x_0.shape[: self.ndim_micro_shape]
        log_mean_coeff = log_mean_coeff.view(*macro_shape, *[1 for _ in range(self.ndim_of_a_sample)])
        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape: Tuple) -> Tensor:
        r"""
        $$
                        \epsilon \sim \mathbfcal{N}(0,1)
                        $$
        """
        return torch.randn(*shape)

    def prior_logp(self, z: torch.Tensor) -> Tensor:
        r"""
        log
        $$
        (2\pi)^{-k/2} \det(\Sigma)^{-1/2} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\mathrm{T} \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
        $$
        where $$\Sigma = I$$ and  $$\mathbf{\mu} = 0$$
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps


class SubVPSDE(SDE):
    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20, n_discretization_steps: int = 1000, ndim_micro_shape: int = 2
    ):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
        ------
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          n_discretization_steps: number of discretization steps
          ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    @override
    def T(self) -> float:
        return 1

    @override
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        macro_shape = x.shape[: self.ndim_micro_shape]
        beta_t = beta_t.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)])
        drift = -0.5 * beta_t * x
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t, mask=None):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        macro_shape = x.shape[: self.ndim_micro_shape]
        log_mean_coeff = log_mean_coeff.view(*macro_shape, *[1 for _ in range(self.ndim_of_a_sample)])
        mean = torch.exp(log_mean_coeff) * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, n_discretization_steps=1000, ndim_micro_shape=2):
        """Construct a Variance Exploding SDE.

        Args:
        ------
          sigma_min: smallest sigma.
          sigma_max: largest sigma.
          n_discretization_steps: number of discretization steps
          ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), n_discretization_steps)
        )

    @property
    @override
    def T(self) -> float:
        return 1

    @override
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
        )
        return drift, diffusion

    @override
    def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        """SMLD(NCSN) discretization."""
        timestep = (t * (self.n_discretization_steps - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0, torch.zeros_like(t), self.discrete_sigmas[timestep - 1].to(t.device)
        )
        f = torch.zeros_like(x)
        g = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, g

    def marginal_prob(self, x, t, mask=None):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (
            2 * self.sigma_max**2
        )
