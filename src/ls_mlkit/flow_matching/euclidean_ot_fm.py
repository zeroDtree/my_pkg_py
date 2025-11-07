from math import e
from typing import Any, Callable

from numpy.f2py.rules import k
import torch
from torch.nn import Module
from torch import Tensor

from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .base_fm import BaseFlow, BaseFlowConfig
from tqdm.auto import tqdm
from .time_scheduler import FlowMatchingTimeScheduler
from .model_interface import Model4FMInterface


@inherit_docstrings
class EuclideanOTFlowConfig(BaseFlowConfig):
    EPS = 1e-5

    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int = 2,
        n_inference_steps: int = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ) -> None:
        super().__init__(
            ndim_micro_shape=ndim_micro_shape,
            n_discretization_steps=n_discretization_steps,
            n_inference_steps=n_inference_steps,
            *args,
            **kwargs,
        )


@inherit_docstrings
class EuclideanOTFlow(BaseFlow):
    def __init__(
        self,
        config: EuclideanOTFlowConfig,
        time_scheduler: FlowMatchingTimeScheduler,
        masker: MaskerInterface,
        model: Module | Model4FMInterface,
        loss_fn: Callable,
    ) -> None:
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.config: EuclideanOTFlowConfig = config
        self.masker: MaskerInterface = masker
        self.model: Module | Model4FMInterface = model
        self.loss_fn = loss_fn

    def compute_loss(self, batch, *args, **kwargs):
        batch = self.model.prepare_batch_data_for_input(batch)
        x_1 = batch["x_1"]
        device = x_1.device
        macro_shape: tuple[int, ...] = self.get_macro_shape(x_1)
        t = batch.get("t", torch.rand(macro_shape, device=device))
        padding_mask: Any | None = batch.get("padding_mask", None)
        copied_t = t.clone().detach()
        t: torch.Tensor = self.complete_micro_shape(t)
        x_0: torch.Tensor = torch.randn_like(x_1, device=device)
        x_t = x_0 * (1 - t) + x_1 * t
        dx_t = x_1 - x_0
        p_dx_t = self.model(x_t=x_t, t=copied_t, padding_mask=padding_mask, *args, **kwargs)
        loss = self.loss_fn(p_dx_t, dx_t, padding_mask)
        return loss

    def step(self, x_t, t, padding_mask=None, *args, **kwargs):
        device = x_t.device
        idx = kwargs.get("idx")
        t_start = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx]
        t_end = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx + 1]
        copied_t_start = t_start.clone().detach()
        copied_t_end = t_end.clone().detach()
        t_start: torch.Tensor = self.complete_micro_shape(copied_t_start)
        t_end: torch.Tensor = self.complete_micro_shape(copied_t_end)

        return x_t + (t_end - t_start) * self.model(
            x_t=x_t
            + self.model(x_t=x_t, t=copied_t_start, padding_mask=padding_mask, *args, **kwargs) * (t_end - t_start) / 2,
            t=copied_t_start + (copied_t_end - copied_t_start) / 2,
            padding_mask=padding_mask,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def sampling(self, shape, device, x_init_posterior=None, *args, **kwargs) -> Tensor:
        x_0 = self.prior_sampling(shape).to(device)
        if x_init_posterior is not None:
            x_0 = x_init_posterior * self.config.EPS + (1 - self.config.EPS) * x_0
        x_t = x_0

        masker = self.masker
        macro_shape = self.get_macro_shape(x_t)

        time_steps = self.time_scheduler.get_discrete_timesteps_schedule().to(device)
        for idx, t in enumerate(tqdm(time_steps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            kwargs["idx"] = idx
            x_t = self.step(x_t=x_t, t=t, padding_mask=no_padding_mask, *args, **kwargs)
        return x_t

    @torch.no_grad()
    def inpainting(
        self,
        x,
        padding_mask,
        inpainting_mask,
        device,
        x_init_posterior=None,
        inpainting_mask_key="inpainting_mask",
        *args,
        **kwargs,
    ) -> Tensor:
        x_1 = x
        shape = x_1.shape
        config = self.config
        masker = self.masker
        macro_shape = shape[: -config.ndim_micro_shape]
        # Add inpainting_mask to kwargs so it gets passed to the model
        kwargs[inpainting_mask_key] = inpainting_mask

        x_0 = self.prior_sampling(shape).to(device)
        if x_init_posterior is not None:
            x_0 = x_init_posterior * self.EPS + (1 - self.EPS) * x_0
        x_t = x_0

        x_1 = masker.apply_mask(x_1, padding_mask)
        timesteps = self.time_scheduler.get_discrete_timesteps_schedule().to(device)
        for idx, t in enumerate(tqdm(timesteps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            x_t = self.recovery_bright_rigion(
                x_known=x_1, x_t=x_t, t=t, padding_mask=padding_mask, inpainting_mask=inpainting_mask, x_prior=x_0
            )
            kwargs["idx"] = idx
            x_t = self.step(x_t=x_t, t=t, padding_mask=padding_mask, *args, **kwargs)
            x_t = masker.apply_mask(x_t, padding_mask)
        x_t = masker.apply_inpainting_mask(x_1, x_t, inpainting_mask)
        return x_t

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(shape)

    def recovery_bright_rigion(self, x_known, x_t, t, padding_mask, inpainting_mask, x_prior) -> Tensor:
        t_start = self.time_scheduler.get_continuous_timesteps_schedule()[t]
        t_start = self.complete_micro_shape(t_start)
        x_1_t = t_start * x_known + (1 - t_start) * x_prior
        x_t = self.masker.apply_inpainting_mask(x_1_t, x_t, inpainting_mask)
        return x_t
