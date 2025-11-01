from copy import deepcopy
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Module
from tqdm.auto import tqdm


class EuclideanFlowConfig:
    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int = 2,
        n_inference_steps: int = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        self.n_discretization_steps = n_discretization_steps
        self.ndim_micro_shape = ndim_micro_shape
        if n_inference_steps is not None:
            self.n_inference_steps = n_inference_steps
        else:
            self.n_inference_steps = n_discretization_steps

    def to(self, device: torch.device | str | Tensor, inplace: bool = True) -> "EuclideanFlowConfig":
        """Move the config to the given device

        Args:
            device (torch.device | str | Tensor): the device to move the config to
            inplace (bool, optional): whether to move the config in place. Defaults to True.

        Returns:
            EuclideanFlowConfig: the config moved to the given device
        """
        obj = self if inplace else deepcopy(self)
        if isinstance(device, Tensor):
            device = device.device
        for k, v in obj.__dict__.items():
            if isinstance(v, Tensor):
                setattr(obj, k, v.to(device))
        return obj


class EuclideanFlow(Module):
    def __init__(self, config: EuclideanFlowConfig, model: Module, loss_fn: Callable):
        super().__init__()
        self.config = config
        self.model = model
        self.loss_fn = loss_fn

    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        """Compute the loss of the flow, used in training.

        Args:
            x_1 (Tensor): (*macro_shape, *micro_shape)
            t (Tensor): (*macro_shape, )
            padding_mask (Tensor): (*macro_shape, ). Defaults to None.

        Returns:
            Tensor: (*macro_shape, )
        """
        x_1 = batch["x_1"]
        t = batch.get("t", None)
        padding_mask = batch.get("padding_mask", None)
        device = x_1.device
        macro_shape = self.get_macro_shape(x_1)
        if t is None:
            t = torch.rand(macro_shape, device=device)
        copied_t = t.clone().detach()
        t = self.complete_micro_shape(t)
        x_0 = torch.randn_like(x_1, device=device)
        x_t = x_0 * (1 - t) + x_1 * t
        dx_t = x_1 - x_0
        p_dx_t = self.model.forward(x_t=x_t, t=copied_t, padding_mask=padding_mask, *args, **kwargs)
        loss = self.loss_fn(p_dx_t, dx_t, padding_mask)
        return loss

    def forward(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        """Module Forward, pytorch

        Args:
            x_t (Tensor): (*macro_shape, *micro_shape)
            t (Tensor): (*macro_shape, )
            padding_mask (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: (*macro_shape, *micro_shape)
        """
        return self.compute_loss(batch, *args, **kwargs)

    @torch.no_grad()
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor, padding_mask: Tensor = None) -> Tensor:
        """Step the flow, used for sampling.

        Args:
            x_t (Tensor): (*macro_shape, *micro_shape)
            t_start (Tensor): (*macro_shape, )
            t_end (Tensor): (*macro_shape, )
            padding_mask (Tensor, optional): (*macro_shape, ). Defaults to None.

        Returns:
            Tensor: (*macro_shape, *micro_shape)
        """
        copied_t_start = t_start.clone().detach()
        copied_t_end = t_end.clone().detach()
        t_start = self.complete_micro_shape(copied_t_start)
        t_end = self.complete_micro_shape(copied_t_end)

        return x_t + (t_end - t_start) * self.model.forward(
            t=copied_t_start + (copied_t_end - copied_t_start) / 2,
            x_t=x_t + self.model.forward(x_t=x_t, t=copied_t_start, padding_mask=padding_mask) * (t_end - t_start) / 2,
            padding_mask=padding_mask,
        )

    def get_macro_shape(self, x: Tensor) -> tuple[int, ...]:
        return x.shape[: -self.config.ndim_micro_shape]

    def get_micro_shape(self, x: Tensor) -> tuple[int, ...]:
        return x.shape[-self.config.ndim_micro_shape :]

    def get_macro_and_micro_shape(self, x: Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return x.shape[: -self.config.ndim_micro_shape], x.shape[-self.config.ndim_micro_shape :]

    def complete_micro_shape(self, x: Tensor) -> Tensor:
        """Complete the micro shape of :math:`x`, assuming the macro shape is already known.

        Args:
            x (Tensor): (*macro_shape,)

        Returns:
            Tensor: (*macro_shape, *micro_shape)
        """
        return x.view(*x.shape, *([1] * self.config.ndim_micro_shape))

    @torch.no_grad()
    def sampling_x1_unconditionally(self, shape, device) -> Tensor:
        """Sample :math:`x_1` unconditionally

        Args:
            shape (tuple): the shape of the sample
            device (device): the device to use for sampling

        Returns:
            Tensor: (*macro_shape, *micro_shape)
        """
        x_t = torch.randn(shape, device=device)
        macro_shape = self.get_macro_shape(x_t)
        timesteps = torch.linspace(0, 1, self.config.n_inference_steps + 1, device=device)
        for i in tqdm(range(len(timesteps) - 1)):
            t_start = timesteps[i] * torch.ones(macro_shape, device=device)
            t_end = timesteps[i + 1] * torch.ones(macro_shape, device=device)
            x_t = self.step(x_t=x_t, t_start=t_start, t_end=t_end)
        return x_t
