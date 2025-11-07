r"""
Base Diffuser Config and Base Diffuser.
"""

import abc
from typing import Any, Tuple

from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_config_class import BaseConfig as BaseConfigClass
from .time_scheduler import TimeScheduler


class BaseDiffuserConfig(BaseConfigClass):
    r"""Diffuser configure base class"""

    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        r"""
        Args:
            n_discretization_steps (``int``): number of discretization steps
            ndim_micro_shape (``int``): umber of dimensions of a sample
        """
        super().__init__(*args, **kwargs)
        self.n_discretization_steps: int = n_discretization_steps
        self.ndim_micro_shape: int = ndim_micro_shape


class BaseDiffuser(Module, abc.ABC):
    r"""
    Base Diffuser Class
    """

    def __init__(
        self,
        config: BaseDiffuserConfig,
        time_scheduler: TimeScheduler,
    ):
        r"""Initialize the BaseDiffuser

        Args:
            config (``BaseDiffuserConfig``): the config of the diffuser
            time_scheduler (``TimeScheduler``): the time scheduler of the diffuser
        """
        abc.ABC.__init__(self)
        Module.__init__(self)
        self.config: BaseDiffuserConfig = config
        self.time_scheduler: TimeScheduler = time_scheduler

    @abc.abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        r"""Sample initial noise used for reverse process

        Args:
            shape (``Tuple[int, ...]``): the shape of the sample

        Returns:
            ``Tensor``: the initial noise
        """

    @abc.abstractmethod
    def forward_process(
        self, x_0: Tensor, discrete_t: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        r"""Forward process, from :math:`x_0` to :math:`x_t`

        Args:
            x_0 (``Tensor``): :math:`x_0`
            discrete_t (``Tensor``): the discrete time steps :math:`t`
            mask (``Tensor``): the mask of the sample

        Returns:
            ``dict``: a dictionary that must contain the key "x_t"
        """

    @abc.abstractmethod
    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> dict:
        r"""Compute loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict``: a dictionary that must contain the key "loss"
        """

    @abc.abstractmethod
    def sample_xtm1_conditional_on_xt(
        self, x_t, discrete_t: Tensor, padding_mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> Tensor:
        r"""Sample :math:`x_{t-1}` conditional on :math:`x_t`

        Args:
            x_t : :math:`x_t`
            discrete_t (``Tensor``): the discrete time steps :math:`t`
            padding_mask (``Tensor``): the padding mask

        Returns:
            ``Tensor``: :math:`x_{t-1}`
        """

    def forward(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]):
        r"""Forward function, input batch of data and return the dictionary containing the loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict``: a dictionary that must contain the key "loss"
        """
        return self.compute_loss(batch, *args, **kwargs)

    def get_macro_and_micro_shape(self, x: Tensor) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        r"""Get the macro and micro shape of :math:`x`

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``Tuple[Tuple[int, ...], Tuple[int, ...]]``: the macro and micro shape of :math:`x`
        """
        ndim_micro_shape = self.config.ndim_micro_shape
        return x.shape[:-ndim_micro_shape], x.shape[-ndim_micro_shape:]

    def get_macro_shape(self, x: Tensor) -> Tuple[int, ...]:
        r"""Get the macro shape of :math:`x`

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``Tuple[int, ...]``: the shape of the macro part of :math:`x`
        """
        return x.shape[: -self.config.ndim_micro_shape]

    def complete_micro_shape(self, x: Tensor) -> Tensor:
        """Complete the micro shape of :math:`x`, assuming the macro shape is already known

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``Tensor``: :math:`x` with the micro shape completed
        """
        return x.view(*x.shape, *([1] * self.config.ndim_micro_shape))
