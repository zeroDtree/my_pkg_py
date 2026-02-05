import abc
import functools

import torch
from overrides import override
from torch import Tensor

from ...util.decorators import register_class_to_dict
from .base_sde import SDE
from .sde_lib import VPSDE

_CORRECTORS = {}

register_corrector = functools.partial(register_class_to_dict, global_dict=_CORRECTORS)


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde: SDE, score_fn: object, snr: float, n_steps: int):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Tensor, mask=None):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """


@register_corrector(key_name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps): ...

    def update_fn(self, x, t, mask=None):
        return x, x


@register_corrector(key_name="langevin_corrector")
class LangevinCorrector(Corrector):
    def __init__(self, sde: SDE, score_fn: object, snr: float, n_steps: int, ndim_micro_shape: int = 3):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

        self.ndim_micro_shape = ndim_micro_shape

    @override
    def update_fn(self, x: Tensor, t: Tensor, mask=None):
        self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        alpha = torch.ones_like(t)

        x_mean = x

        for _ in range(n_steps):
            grad = score_fn(x, t, mask)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size.view(step_size.shape[0], *[1 for _ in range(self.ndim_micro_shape)]) * grad
            x = (
                x_mean
                + torch.sqrt(step_size * 2).view(step_size.shape[0], *[1 for _ in range(self.ndim_micro_shape)]) * noise
            )

        return x, x_mean
