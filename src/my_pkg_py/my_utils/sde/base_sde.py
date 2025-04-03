"""Abstract SDE classes"""

import abc
import torch
from torch import Tensor
from typing import Tuple


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, n_discretization_steps: int):
        """
        n_discretization_steps: number of discretization time steps.
        """
        super().__init__()
        self.n_discretization_steps = n_discretization_steps

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """returns  drift and diffusion coefficient of the SDE"""
        pass

    def get_discretized_drift_and_diffusion(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Discretize the SDE in the form: 
		$$
        \begin{align*}
        dx &= f(x, t)dt + g(x,t)d z\\
        x_{t+\Delta t} &= x_t + f(x_t, t)(\Delta t) + g(x_t, t) \epsilon, \epsilon \sim \mathcal{N}(0,|\Delta t|))\\
        \end{align*}
        $$
        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)
          
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
        drift, diffusion = self.get_drift_and_diffusion(x, t)
        f = drift * dt
        g = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, g

    def get_reverse_sde(self, score_fn: object, use_probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          use_probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        n_discretization_steps = self.n_discretization_steps
        T = self.T
        get_forward_drift_and_diffusion = self.get_drift_and_diffusion
        get_forward_discretized_drift_and_diffusion = self.get_discretized_drift_and_diffusion

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.n_discretization_steps = n_discretization_steps
                self.use_probability_flow = use_probability_flow

            @property
            def T(self):
                return T

            def get_drift_and_diffusion(self, x, t):
                """
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
                drift, diffusion = get_forward_drift_and_diffusion(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.0)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def get_discretized_drift_and_diffusion(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler.
                $$
				\begin{align*}
					rev\_f &= (f(x,t) - g(x,t)^2 \nabla_x \log p_t(x)) |\Delta t| \\
					rev\_g &= g(x,t) \sqrt{|\Delta t|}
				\end{align*}
				$$
                """
                f, g = get_forward_discretized_drift_and_diffusion(x, t)
                rev_f = f - g[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.0)
                rev_g = torch.zeros_like(g) if self.probability_flow else g
                return rev_f, rev_g

        return RSDE()
