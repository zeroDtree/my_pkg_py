from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from .base_sde import SDE


class VPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, ndim_micro_shape: int = 2):
        r"""Construct a Variance Preserving SDE.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(ndim_micro_shape=ndim_micro_shape)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self) -> float:
        return 1

    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""continuous DDPM SDE

        .. math::

            dx &= -\frac{1}{2}\beta_t x dt + \sqrt{\beta_t} dw

        Args:
            x:
            t: (macro_shape)
            mask:

        Returns:
            drift: shape = x.shape
            diffusion: shape=x.macro_shape
        """
        macro_shape = x.shape[: -self.ndim_micro_shape]
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)]) * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def get_score(self, x_t, mean, std) -> Tensor:
        r"""

        .. math::

            p_{0t} (x_t|x_0) = \nabla_{x_t} \ln p_{0t} (x_t|x_0)

        """
        score = -(x_t - mean) / std**2
        return score

    def get_a_b(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """x_t = a * x_0 + b * epsilon, epsilon ~ N(0, 1)

        Args:
            t (``Tensor``): continuous time

        Returns:
            ``Tuple[Tensor, Tensor]``: a, b
        """
        macro_shape = t.shape
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0  # mcro_shape
        log_mean_coeff = log_mean_coeff.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)])
        a = torch.exp(log_mean_coeff)
        b = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return a, b

    def forward_process(self, x_0: Tensor, t: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""

        .. math::

            p_{0t} (x_t|x_0)

        .. math::

            \gamma = -\frac{1}{4}t^2 (\beta_1 - \beta_0) - \frac{1}{2} t  \beta_0

            mean = e^{\gamma} * x

            std = \sqrt{1 - e^{2 \gamma }}

        """
        a, b = self.get_a_b(t)
        mean = a * x_0
        x_t = mean + b * torch.randn_like(x_0)
        return {
            "x_t": x_t,
            "mean": mean,
            "std": b,
            "a": a,
            "b": b,
        }

    def forward_from_t1_to_t2(self, x_t1: Tensor, t1: Tensor, t2: Tensor) -> Tensor:
        assert (t1 <= t2).all(), "t1 must be less than or equal to t2"
        a1, b1 = self.get_a_b(t1)
        a2, b2 = self.get_a_b(t2)
        a12 = a2 / a1
        b12 = a2 * torch.sqrt((b2 / a2) ** 2 - (b1 / a1) ** 2)
        x_t2 = a12 * x_t1 + b12 * torch.randn_like(x_t1)
        return x_t2

    def prior_sampling(self, shape: Tuple) -> Tensor:
        r"""
        .. math::
            \epsilon \sim \mathbfcal{N}(0,1)

        """
        return torch.randn(*shape)

    def prior_logp(self, z: torch.Tensor) -> Tensor:
        r"""

        .. math::

            (2\pi)^{-k/2} \det(\Sigma)^{-1/2} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\mathrm{T} \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)

        where :math:`\Sigma = I` and  :math:`\mathbf{\mu} = 0`
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps


class SubVPSDE(SDE):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20, ndim_micro_shape: int = 2):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
            beta_min: value of beta(0)
            beta_max: value of beta(1)
            n_discretization_steps: number of discretization steps
            ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(ndim_micro_shape=ndim_micro_shape)
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    @property
    def T(self) -> float:
        return 1

    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        macro_shape = x.shape[: -self.ndim_micro_shape]
        beta_t = beta_t.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)])
        drift = -0.5 * beta_t * x
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t, mask=None):
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        macro_shape = x.shape[: -self.ndim_micro_shape]
        log_mean_coeff = log_mean_coeff.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)])
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
    def __init__(
        self, sigma_min=0.01, sigma_max=50, n_discretization_steps=1000, ndim_micro_shape=2, drop_first_step=False
    ):
        """Construct a Variance Exploding SDE.

        Args:

            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            n_discretization_steps: number of discretization steps
            ndim_micro_shape: number of dimensions of a sample
        """
        super().__init__(n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.drop_first_step = drop_first_step
        sigma_min = torch.tensor(sigma_min)
        sigma_max = torch.tensor(sigma_max)
        if drop_first_step:
            self.discrete_sigmas = (
                10 ** torch.linspace(torch.log10(sigma_min), torch.log10(sigma_max), n_discretization_steps + 1)[1:]
            )
        else:
            self.discrete_sigmas = torch.exp(
                torch.linspace(torch.log(sigma_min), torch.log(sigma_max), n_discretization_steps)
            )

    @property
    def T(self) -> float:
        return 1

    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""
        .. math::

            dx = 0 dt + \sigma_{min} \left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t \sqrt{2 \log(\frac{\sigma_{max}}{\sigma_{min}})} dw
            \sigma_t = \sigma_{min} \left(\frac{\sigma_{max}}{\sigma_{min}}\right)^t

            diffusion = \sigma_t * \sqrt{2 \log(\frac{\sigma_{max}}{\sigma_{min}})}

        """
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
        )
        return drift, diffusion

    def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""SMLD(NCSN) discretization.
        .. math::

            x_t &= x_0 + g \epsilon

            x_t &\sim \mathcal{N}(x_0, \sigma_t^2)

            \sigma_t^2 &= \sigma_{t-1}^2 + g^2

            g &= \sqrt{\sigma_t^2 - \sigma_{t-1}^2}

        """
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
