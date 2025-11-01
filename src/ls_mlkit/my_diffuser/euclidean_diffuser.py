from typing import Any, Literal, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ls_mlkit.my_diffuser.time_scheduler import TimeScheduler
from ls_mlkit.my_utils.decorators import inherit_docstrings
from ls_mlkit.my_utils.mask.masker_interface import MaskerInterface

from .base_diffuser import BaseDiffuser, BaseDiffuserConfig
from .conditioner import Conditioner


@inherit_docstrings
class EuclideanDiffuserConfig(BaseDiffuserConfig):
    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape, *args, **kwargs
        )


@inherit_docstrings
class EuclideanDiffuser(BaseDiffuser):
    def __init__(
        self,
        config: EuclideanDiffuserConfig,
        time_scheduler: TimeScheduler,
        masker: MaskerInterface,
        conditioner_list: list[Conditioner] = [],
    ):
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.masker = masker
        self.conditioner_list = conditioner_list

    def forward_process_one_step(self, x: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Forward process one step

        Args:
            x (Tensor): the sample
            t (Tensor): the timestep
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the sample at the next timestep
        """

    def forward_process_n_step(
        self, x: Tensor, t: Tensor, next_t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Forward process n step, from t to next_t

        Args:
            x (Tensor): the sample
            t (Tensor): the timestep
            next_t (Tensor): the next timestep
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the sample at the next timestep
        """

    def get_accumulated_conditional_score(
        self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        r"""Get the accumulated conditional score

        Args:
            x_t (Tensor): :math:`x_t`
            t (Tensor): :math:`t`
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the accumulated conditional score
        """
        accumulated_conditional_score = torch.zeros_like(x_t)
        for conditioner in self.conditioner_list:
            accumulated_conditional_score += conditioner.get_conditional_score(x_t, t, padding_mask, *args, **kwargs)
        return accumulated_conditional_score

    @torch.no_grad()
    def sample_x0_unconditionally(
        self, shape: Tuple[int, ...], device, x_init_posterior: Tensor = None, *args: Any, **kwargs: Any
    ) -> Tensor:
        r"""Sample :math:`x_0` unconditionally

        Args:
            shape (Tuple[int, ...]): the shape of the sample
            device (device): the device to use for sampling
            x_init_posterior (Tensor): Use x_init_posterior as the initial posterior if provided, otherwise sample from prior.
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            Tensor: :math:`x_0`
        """
        config = self.config
        if x_init_posterior is not None:
            shape = x_init_posterior.shape
        macro_shape = shape[: -self.config.ndim_micro_shape]
        masker = self.masker
        if x_init_posterior is None:
            x_t = self.prior_sampling(shape).to(device)
        else:
            x_t = x_init_posterior
            padding_mask = kwargs.get("padding_mask", None)
            if padding_mask is None:
                padding_mask = masker.get_full_bright_mask(x_t)
            x_t = self.forward_process(
                x_t,
                torch.ones(macro_shape, device=device, dtype=torch.long) * (config.n_discretization_steps - 1),
                padding_mask,
            )["x_t"]

        # Get timesteps from the specific diffuser implementation
        timesteps = self.time_scheduler.get_discrete_timesteps_schedule()
        for t in tqdm(timesteps):
            t = torch.ones(macro_shape, device=device) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            x_t = self.sample_xtm1_conditional_on_xt(x_t, t, no_padding_mask, *args, **kwargs)

        return x_t

    @torch.no_grad()
    def inpainting_x0_unconditionally(
        self,
        x_0: Tensor,
        padding_mask: Tensor,
        inpainting_mask: Tensor,
        device,
        recovery_mode: Literal["x_0", "x_t"] = "x_t",
        n_repaint_steps: int = 1,
        x_init_posterior: Tensor = None,
        inpainting_mask_key="inpainting_mask",
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Inpaint :math:`x_0` unconditionally

        Args:
            x_0 (Tensor): :math:`x_0`
            padding_mask (Tensor): the padding mask
            inpainting_mask (Tensor): the inpainting mask
            device (device): the device to use for sampling
            recovery_mode (Literal["x_0", "x_t"]): the recovery mode
            n_repaint_steps (int): the number of repaint steps, n_repaint_steps > 1 means use RePaint.
            x_init_posterior (Tensor): Use x_init_posterior as the initial posterior if provided, otherwise sample from prior.
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            Tensor: :math:`x_0` inpainted
        """
        shape = x_0.shape
        config = self.config
        macro_shape = shape[: -config.ndim_micro_shape]
        masker = self.masker
        x_t = None
        if x_init_posterior is None:
            x_t = self.prior_sampling(shape).to(device)
        else:
            x_t = x_init_posterior
            x_t = self.forward_process(
                x_t,
                torch.ones(macro_shape, device=device, dtype=torch.long) * (config.n_discretization_steps - 1),
                padding_mask,
            )["x_t"]
        x_0 = masker.apply_mask(x_0, padding_mask)
        timesteps = self.time_scheduler.get_discrete_timesteps_schedule()

        # Add inpainting_mask to kwargs so it gets passed to the model
        kwargs[inpainting_mask_key] = inpainting_mask

        for i, t in enumerate(tqdm(timesteps)):
            for u in range(1, n_repaint_steps + 1):
                t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
                if recovery_mode == "x_t":
                    x_0t = self.forward_process(x_0, t, padding_mask)["x_t"]
                else:
                    x_0t = x_0
                x_t = masker.apply_inpainting_mask(x_0t, x_t, inpainting_mask)
                x_t = self.sample_xtm1_conditional_on_xt(x_t, t, padding_mask, *args, **kwargs)  # get x_tm1
                x_t = masker.apply_mask(x_t, padding_mask)
                if u < n_repaint_steps and (t > 0).all():
                    assert i < len(timesteps) - 1
                    prev_t = timesteps[i + 1].to(device)
                    x_t = self.forward_process_n_step(x_t, prev_t, t, padding_mask, *args, **kwargs)
        x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)
        return x_t
