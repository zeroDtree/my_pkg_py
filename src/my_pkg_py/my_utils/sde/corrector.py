import abc
import torch
from torch import Tensor
from overrides import override
from .base_sde import SDE
from .sde_lib import VPSDE, VESDE, SubVPSDE
from my_utils.decorators import register_class_to_dict
import functools

_CORRECTORS = {}

register_corrector: function = functools.partial(register_class_to_dict, global_dict=_CORRECTORS)


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde: SDE, score_fn: object, snr: float, n_steps: int):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Tensor):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps): ...

    def update_fn(self, x, t):
        return x, x


@register_corrector(key_name="langevin_corrector")
class LangevinCorrector(Corrector):
    def __init__(self, sde: SDE, score_fn: object, snr: float, n_steps: int):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE) and not isinstance(sde, SubVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    @override
    def update_fn(self, x: Tensor, t: Tensor):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
            timestep = (t * (sde.n_discretization_steps - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean
