from typing import Any, Tuple

import torch
from torch import Tensor
from tqdm.auto import tqdm

from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .base_diffuser import BaseDiffuser, BaseDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler
import numpy as np
from ..util.base_class.base_gm_class import GMHookStageType, GMHookManager, GMHook, GMHookHandler


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
        if x_init_posterior is not None:
            shape = x_init_posterior.shape
        macro_shape = shape[: -self.config.ndim_micro_shape]
        macro_shape = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_GET_MACRO_SHAPE,
            tgt_key_name="macro_shape",
            macro_shape=macro_shape,
            batch=kwargs,
        )
        masker = self.masker
        if x_init_posterior is None:
            x_t = self.prior_sampling(shape).to(device)
        else:
            padding_mask = kwargs.get("padding_mask", None)
            if padding_mask is None:
                padding_mask = masker.get_full_bright_mask(x_t)
            t_a = torch.ones(macro_shape, device=device, dtype=torch.long) * (
                self.time_scheduler.get_timestep_index_start() - 1
            )
            t_a = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_a, batch=kwargs
            )
            t_a = self.complete_micro_shape(t_a)
            t_b = (
                torch.ones(macro_shape, device=device, dtype=torch.long) * self.time_scheduler.get_timestep_index_end()
            )
            t_b = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_b, batch=kwargs
            )
            t_b = self.complete_micro_shape(t_b)

            x_t = self.forward_process(
                x_init_posterior,
                t_a,
                t_b,
                padding_mask,
                is_continuous_time=False,
            )["x_t"]

        x_list = [x_t]
        E_x0_xt_list = [x_t]

        time_steps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for idx, t in enumerate(tqdm(time_steps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            t = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t, batch=kwargs
            )
            t = self.complete_micro_shape(t)
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
        self.config = self.config.to(device)
        x_0 = x
        shape = x_0.shape
        macro_shape = shape[: -self.config.ndim_micro_shape]
        # >>>>>>>>>>>>>>>>>>>
        macro_shape = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_GET_MACRO_SHAPE,
            tgt_key_name="macro_shape",
            macro_shape=macro_shape,
            batch=kwargs,
        )
        # <<<<<<<<<<<<<<<<<<
        masker = self.masker
        # Add inpainting_mask to kwargs so it gets passed to the model
        kwargs[inpainting_mask_key] = inpainting_mask

        x_t = None
        if x_init_posterior is None:
            x_t = self.prior_sampling(shape).to(device)
        else:
            t_a = torch.ones(macro_shape, device=device, dtype=torch.long) * (
                self.time_scheduler.get_timestep_index_start() - 1
            )
            t_a = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_a, batch=kwargs
            )
            t_a = self.complete_micro_shape(t_a)
            t_b = (
                torch.ones(macro_shape, device=device, dtype=torch.long) * self.time_scheduler.get_timestep_index_end()
            )
            t_b = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_b, batch=kwargs
            )
            t_b = self.complete_micro_shape(t_b)

            x_t = self.forward_process(
                x_init_posterior,
                t_a,
                t_b,
                padding_mask,
                is_continuous_time=False,
            )["x_t"]
        x_0 = masker.apply_mask(x_0, padding_mask)
        x_T = x_t.detach().clone()

        x_list = [x_t]
        E_x0_xt_list = [x_t]

        timesteps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            t = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t, batch=kwargs
            )
            t = self.complete_micro_shape(t)
            for u in range(1, n_repaint_steps + 1):
                x_t = self.recover_bright_region(
                    x_known=x_0,
                    x_t=x_t,
                    t=t,
                    padding_mask=padding_mask,
                    x_prior=x_T,
                    **kwargs,
                )
                step_output = self.step(x_t, t, padding_mask, *args, **kwargs)  # get x_tm1
                x_t = step_output["x"]
                if "E_x0_xt" in step_output:
                    E_x0_xt_list.append(step_output["E_x0_xt"])
                x_t = masker.apply_mask(x_t, padding_mask)
                if u < n_repaint_steps and (t > 0).all():
                    prev_t = timesteps[i + 1].to(device)
                    prev_t = self.hook_manager.run_hooks(
                        stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=prev_t, batch=kwargs
                    )
                    prev_t = self.complete_micro_shape(prev_t)
                    x_t = self.forward_process(x_t, prev_t, t, padding_mask, is_continuous_time=False, *args, **kwargs)["x_t"]
            if return_all:
                x_list.append(x_t)
        x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)

        return {"x": x_t, "x_list": x_list, "E_x0_xt_list": E_x0_xt_list}

    def recover_bright_region(self, x_known, x_t, t, padding_mask, inpainting_mask, x_prior, *args, **kwargs) -> Tensor:
        x_0 = x_known
        t_a = torch.ones_like(t, device=t.device) * (self.time_scheduler.get_timestep_index_start() - 1)
        t_a = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_a, batch=kwargs
        )
        x_0t = self.forward_process(
            x_0,
            t_a,
            t,
            padding_mask,
            is_continuous_time=False,
        )["x_t"]
        x_t = self.masker.apply_inpainting_mask(x_0t, x_t, inpainting_mask)
        return x_t

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://huggingface.co/papers/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)  # (batch_size, 1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample
