from typing import Any, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .base_diffuser import BaseDiffuser, BaseDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class EuclideanDiffuserConfig(BaseDiffuserConfig):
    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        n_inference_steps: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
            n_inference_steps=n_inference_steps,
            *args,
            **kwargs,
        )


@inherit_docstrings
class EuclideanDiffuser(BaseDiffuser):
    def __init__(
        self,
        config: EuclideanDiffuserConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: MaskerInterface,
    ):
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.config: EuclideanDiffuserConfig = config
        self.time_scheduler: DiffusionTimeScheduler = time_scheduler
        self.masker = masker

    def forward_process_n_step(
        self, x: Tensor, t: Tensor, next_t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        r"""Forward process n step, from t to next_t

        Args:
            x (``Tensor``): the sample
            t (``Tensor``): the timestep
            next_t (``Tensor``): the next timestep
            padding_mask (``Tensor``): the padding mask

        Returns:
            ``Tensor``: the sample at the next timestep
        """

    @torch.no_grad()
    def sampling(
        self,
        shape: Tuple[int, ...],
        device,
        x_init_posterior: Tensor = None,
        return_all=False,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
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

        x_list = [x_t]
        E_x0_xt_list = [x_t]

        time_steps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for idx, t in enumerate(tqdm(time_steps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            kwargs["idx"] = idx
            step_output = self.step(x_t=x_t, t=t, padding_mask=no_padding_mask, *args, **kwargs)
            x_t = step_output["x"]
            if "E_x0_xt" in step_output:
                E_x0_xt_list.append(step_output["E_x0_xt"])
            if return_all:
                x_list.append(x_t)
        return {"x": x_t, "x_list": x_list, "E_x0_xt_list": E_x0_xt_list}

    @torch.no_grad()
    def inpainting(
        self,
        x: Tensor,
        padding_mask: Tensor,
        inpainting_mask: Tensor,
        device,
        x_init_posterior: Tensor = None,
        inpainting_mask_key="inpainting_mask",
        n_repaint_steps: int = 1,
        return_all=False,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        x_0 = x
        shape = x_0.shape
        config = self.config
        macro_shape = shape[: -config.ndim_micro_shape]
        masker = self.masker
        # Add inpainting_mask to kwargs so it gets passed to the model
        kwargs[inpainting_mask_key] = inpainting_mask

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
        x_T = x_t.detach().clone()

        x_list = [x_t]
        E_x0_xt_list = [x_t]

        timesteps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for i, t in enumerate(tqdm(timesteps)):
            for u in range(1, n_repaint_steps + 1):
                t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
                x_t = self.recover_bright_region(
                    x_known=x_0, x_t=x_t, t=t, inpainting_mask=inpainting_mask, padding_mask=padding_mask, x_prior=x_T
                )
                step_output = self.step(x_t, t, padding_mask, *args, **kwargs)  # get x_tm1
                x_t = step_output["x"]
                if "E_x0_xt" in step_output:
                    E_x0_xt_list.append(step_output["E_x0_xt"])
                x_t = masker.apply_mask(x_t, padding_mask)
                if u < n_repaint_steps and (t > 0).all():
                    assert i < len(timesteps) - 1
                    prev_t = timesteps[i + 1].to(device)
                    x_t = self.forward_process_n_step(x_t, prev_t, t, padding_mask, *args, **kwargs)
            if return_all:
                x_list.append(x_t)
        x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)

        return {"x": x_t, "x_list": x_list, "E_x0_xt_list": E_x0_xt_list}

    def recover_bright_region(self, x_known, x_t, t, padding_mask, inpainting_mask, x_prior) -> Tensor:
        x_0 = x_known
        x_0t = self.forward_process(x_0, t, padding_mask)["x_t"]
        x_t = self.masker.apply_inpainting_mask(x_0t, x_t, inpainting_mask)
        return x_t
