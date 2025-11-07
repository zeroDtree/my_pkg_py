from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor

from ..decorators import inherit_docstrings
from .base_loss_class import BaseLossClass, BaseLossConfig


@inherit_docstrings
class BaseGenerativeModelConfig(BaseLossConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: int = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        """
        Args:
            ndim_micro_shape (``int``): number of dimensions of a sample
            n_discretization_steps (``int``): number of discretization steps
            n_inference_steps (``int``, *optional*): number of inference steps
        """
        super().__init__(ndim_micro_shape=ndim_micro_shape, *args, **kwargs)
        self.n_discretization_steps: int = n_discretization_steps
        if n_inference_steps is not None:
            self.n_inference_steps: int = n_inference_steps
        else:
            self.n_inference_steps: int = n_discretization_steps


@inherit_docstrings
class BaseGenerativeModel(BaseLossClass):
    """
    abstract method: compute_loss, step, sampling, inpainting
    """

    def __init__(self, config: BaseGenerativeModelConfig):
        super().__init__(config=config)
        self.config: BaseGenerativeModelConfig = config

    @abstractmethod
    def prior_sampling(self, shape: tuple[int, ...]) -> Tensor:
        r"""prior sampling

        Args:
            shape (``tuple[int, ...]``): the shape of the sample

        Returns:
            ``Tensor``: data from prior distribution
        """

    @abstractmethod
    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ) -> Tensor:
        """_summary_

        Args:
            x_t (``Tensor``): _description_
            t (``Tensor``): t is discrete timestep
            padding_mask (``Tensor``, *optional*): _description_. Defaults to None.

        Returns:
            ``Tensor``: _description_
        """

    @abstractmethod
    def sampling(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        x_init_posterior=None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ) -> Tensor:
        """Sample unconditionally

        Args:
            shape (``tuple[int, ...]``): the shape of the sample
            device (``device``): the device to use for sampling
            x_init_posterior (``Tensor``, *optional*): ``(*macro_shape, *micro_shape)``. Defaults to None.
        Returns:
            ``Tensor``: ``(*macro_shape, *micro_shape)``
        """

    @abstractmethod
    def inpainting(
        self,
        x: Tensor,
        padding_mask: Tensor,
        inpainting_mask: Tensor,
        device: torch.device,
        x_init_posterior: Tensor = None,
        inpainting_mask_key="inpainting_mask",
        *args,
        **kwargs,
    ) -> Tensor:
        """Inpainting

        Args:
            x_1 (``Tensor``): ``(*macro_shape, *micro_shape)``
            padding_mask (``Tensor``):
            inpainting_mask (``Tensor``):
            device (``torch.device``): the device to use for sampling
            x_init_posterior (``Tensor``, *optional*): ``(*macro_shape, *micro_shape)``. Defaults to None.
            inpainting_mask_key (``str``, *optional*): the key of the inpainting mask. Defaults to "inpainting_mask".

        Returns:
            ``Tensor``:
        """
