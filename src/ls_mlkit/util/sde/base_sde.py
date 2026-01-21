r"""Abstract SDE classes

Note:
    ``t`` is always continous time step in this module.
"""

import abc
from typing import Tuple

from torch import Tensor


class SDE(abc.ABC):
    r"""
    SDE abstract class. Functions are designed for a mini-batch of inputs.
    """

    def __init__(self, ndim_micro_shape: int = 2):
        r"""Initialize the SDE

        Args:
            ndim_micro_shape (``int``, *optional*): number of dimensions of a sample.
            e.g. for image with shape ``[b, c, h, w]``, ndim_micro_shape = 3
            e.g. for protein with shape ``[b, n_res, 3]``, ndim_micro_shape = 2
        """
        super().__init__()
        self.ndim_micro_shape = ndim_micro_shape

    @property
    @abc.abstractmethod
    def T(self) -> float:
        r"""End time of the SDE."""

    @abc.abstractmethod
    def get_drift_and_diffusion(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""Get the drift and diffusion of the SDE.

        Args:
            x (``Tensor``): the sample.
            t (``Tensor``): the time step.
            mask (``Tensor``, *optional*): the mask of the sample. Defaults to None.

        Returns:
            ``Tuple[Tensor, Tensor]``: the drift and diffusion of the SDE.
        """

    def get_reverse_sde(self, score=None, score_fn: object = None, use_probability_flow=False):
        r"""Create the reverse-time SDE/ODE.

        Args:
            score_fn: A time-dependent score-based model that takes (x ,t, mask) and returns the score.
            use_probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        T = self.T
        ndim_micro_shape = self.ndim_micro_shape
        get_forward_drift_and_diffusion = self.get_drift_and_diffusion
        # get_forward_discretized_drift_and_diffusion = self.get_discretized_drift_and_diffusion

        class RSDE(self.__class__):
            def __init__(self):
                self.use_probability_flow = use_probability_flow
                self.ndim_micro_shape = ndim_micro_shape

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
                nonlocal score, score_fn
                assert score is not None or score_fn is not None, "either score or score_fn must be provided"
                if score is None:
                    score = score_fn(x, t, mask)
                drift, diffusion = get_forward_drift_and_diffusion(x, t, mask=mask)
                rev_diffusion = 0.0 if self.use_probability_flow else diffusion
                diffusion = diffusion.view(
                    *x.shape[: -self.ndim_micro_shape], *[1 for _ in range(self.ndim_micro_shape)]
                )
                rev_drift = drift - diffusion**2 * score * (0.5 if self.use_probability_flow else 1.0)
                # Set the diffusion function to zero for ODEs.
                return rev_drift, rev_diffusion

        return RSDE()
