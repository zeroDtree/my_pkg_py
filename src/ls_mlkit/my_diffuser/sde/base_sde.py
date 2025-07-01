"""Abstract SDE classes"""

import abc
import torch
from torch import Tensor
from typing import Tuple


class SDE(abc.ABC):
    """
    SDE abstract class. Functions are designed for a mini-batch of inputs.
    """

    def __init__(self, n_discretization_steps: int, ndim_micro_shape: int = 2):
        """
        Args:
        ------
        n_discretization_steps: number of discretization time steps.
        ndim_micro_shape: number of dimensions of a sample.
            e.g. for image with shape [b, c, h, w], ndim_micro_shape = 3
            e.g. for protein with shape [b, n_res, 3], ndim_micro_shape = 2


        """
        super().__init__()
        self.n_discretization_steps = n_discretization_steps
        self.ndim_micro_shape = ndim_micro_shape

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""
        pass

    def get_discretization_steps(self, t: Tensor) -> Tensor:
        """Get the discretization steps."""
        return t * (self.n_discretization_steps - 1) / self.T

    def get_diffusion_coefficient_with_proper_shape(self, x: Tensor, diffusion: Tensor) -> Tensor:
        """Get the diffusion coefficient with the proper shape."""
        macro_shape = x.shape[: self.ndim_micro_shape]
        if macro_shape == diffusion.shape:
            return diffusion.view(*macro_shape, *[1 for _ in range(self.ndim_micro_shape)])
        else:
            raise ValueError("No implementation for macro_shape != diffusion.shape")

    def get_target_score(self, x_0: Tensor, x_t: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        r"""Get the target score.
        Returns:
        $$
        \nabla_{x_t} \ln p_{0t} (x_t|x_0)
        $$
        """
        pass

    @abc.abstractmethod
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        """
        returns
        ------
        drift: drift.shape = x.shape
        diffusion: diffusion.shape = x.macro_shape
            General case diffusion.shape =  (*x.macro_shape, d, d) is not implemented.
        """
        pass

    def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""Euler-Maruyama discretization.
		$$
        \begin{align*}
        dx &= f(x, t)dt + g(x,t)d z\\
        x_{t+\Delta t} &= x_t + f(x_t, t)(\Delta t) + g(x_t, t) \epsilon, \epsilon \sim \mathcal{N}(0,|\Delta t|))\\
        \end{align*}
        $$
        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)
          mask: 1 indicates valid region, 0 indicates invalid region
          
          Note: Here dt always greater than 0.
        Returns:
          f, g
          $$
		  \begin{align*}
		  f &= f(x,t) |\Delta t| \\
		  g &= g(x,t) \sqrt{|\Delta t|}
		  \end{align*}
		  $$
        """
        dt = self.T / self.n_discretization_steps
        drift, diffusion = self.get_drift_and_diffusion(x, t, mask=mask)
        f = drift * dt
        diffusion = self.get_diffusion_coefficient_with_proper_shape(x, diffusion)
        g = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, g

    def get_reverse_sde(self, score_fn: object, use_probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes (x ,t, mask) and returns the score.
          use_probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        n_discretization_steps = self.n_discretization_steps
        T = self.T
        ndim_micro_shape = self.ndim_micro_shape
        get_forward_drift_and_diffusion = self.get_drift_and_diffusion
        get_forward_discretized_drift_and_diffusion = self.get_discretized_drift_and_diffusion

        class RSDE(self.__class__):
            def __init__(self):
                self.n_discretization_steps = n_discretization_steps
                self.use_probability_flow = use_probability_flow
                self.ndim_micro_shape = ndim_micro_shape

            @property
            def T(self) -> float:
                return T

            def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
                r"""
                Create the drift and diffusion functions for the reverse SDE/ODE.
                $$
                \begin{align*}
                    dx = (f(x,t) - g(x,t)^2 \nabla_x \log p_t(x)) dt + g(x,t) dw
                \end{align*}
                $$
                if use ODE probability flow:
                $$
                \begin{align*}
                    dx = (f(x,t) - \frac{1}{2} g(x,t)^2 \nabla_x \log p_t(x)) dt
                \end{align*}
                $$
                """
                drift, diffusion = get_forward_drift_and_diffusion(x, t, mask=mask)
                score = score_fn(x, t, mask)
                diffusion = self.get_diffusion_coefficient_with_proper_shape(x, diffusion)
                drift = drift - diffusion**2 * score * (0.5 if self.use_probability_flow else 1.0)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.use_probability_flow else diffusion
                return drift, diffusion

            def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
                r"""Create discretized iteration rules for the reverse diffusion sampler.
                $$
				\begin{align*}
					rev\_f &= (f(x,t) - g(x,t)^2 \nabla_x \log p_t(x)) |\Delta t| \\
					rev\_g &= g(x,t) \sqrt{|\Delta t|}
				\end{align*}
				$$
                """
                f, g = get_forward_discretized_drift_and_diffusion(x, t, mask=mask)
                rev_f = f - g**2 * score_fn(x, t) * (0.5 if self.use_probability_flow else 1.0)
                rev_g = torch.zeros_like(g) if self.use_probability_flow else g
                return rev_f, rev_g

        return RSDE()
