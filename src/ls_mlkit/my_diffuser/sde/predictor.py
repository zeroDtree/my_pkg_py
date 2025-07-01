import abc
from .base_sde import SDE
import torch
from torch import Tensor
from overrides import override
from typing import Tuple
from ls_mlkit.my_utils.decorators import register_class_to_dict
import functools

_PREDICTORS = {}

register_predictor = functools.partial(register_class_to_dict, global_dict=_PREDICTORS)


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde: SDE, score_fn: object, use_probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.get_reverse_sde(score_fn=score_fn, use_probability_flow=use_probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(key_name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, use_probability_flow=False): ...

    def update_fn(self, x, t, mask=None):
        return x, x


@register_predictor(key_name="reverse_diffusion_predictor")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde: SDE, score_fn, use_probability_flow=False, n_dim: int = 3):
        super().__init__(sde=sde, score_fn=score_fn, use_probability_flow=use_probability_flow)
        self.n_dim = n_dim

    @override
    def update_fn(self, x: Tensor, t: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        r"""
        $$
        \begin{align*}
        	x_{t+\Delta t} &= x_t + f(x_t, t)(\Delta t) + g(x_t, t) \epsilon, \epsilon \sim \mathcal{N}(0,\sqrt{\Delta t}))\\
            f &= f(x_t, t)|\Delta t|\\
            g &= g(x_t, t)\sqrt{|\Delta t|}\\
        \end{align*}
        $$
        """
        f, g = self.rsde.get_discretized_drift_and_diffusion(x, t, mask=mask)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + g * z
        return x, x_mean
