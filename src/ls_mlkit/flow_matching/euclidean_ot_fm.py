from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Module
from tqdm.auto import tqdm

from ..diffusion.conditioner.conditioner import Conditioner
from ..diffusion.conditioner.utils import get_accumulated_conditional_score
from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .base_fm import BaseFlow, BaseFlowConfig
from .time_scheduler import FlowMatchingTimeScheduler

EPS = 1e-5


@inherit_docstrings
class EuclideanOTFlowConfig(BaseFlowConfig):

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
        model: Module,
        loss_fn: Callable,
    ) -> None:
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.config: EuclideanOTFlowConfig = config
        self.masker: MaskerInterface = masker
        self.model: Module = model
        self.loss_fn = loss_fn

    def compute_loss(self, **batch):
        x_1 = batch["gt_data"]
        device = x_1.device
        macro_shape: tuple[int, ...] = self.get_macro_shape(x_1)
        t = batch.get("t", torch.rand(macro_shape, device=device))
        padding_mask: Any | None = batch.get("padding_mask", None)
        copied_t = t.clone().detach()
        t: torch.Tensor = self.complete_micro_shape(t)
        x_0: torch.Tensor = torch.randn_like(x_1, device=device)
        x_t = x_0 * (1 - t) + x_1 * t
        dx_t = x_1 - x_0
        model_input_dict = batch
        model_input_dict.pop("gt_data")
        model_input_dict.pop("padding_mask")
        model_input_dict.pop("t", None)
        p_dx_t = self.model(x_t=x_t, t=copied_t, padding_mask=padding_mask, **model_input_dict)["x"]
        loss = self.loss_fn(p_dx_t, dx_t, padding_mask)
        result = {
            "loss": loss,
            "x_1": x_1,
            "x_t": x_t,
            "x_0": x_0,
            "t": t,
            "p_dx_t": p_dx_t,
            "padding_mask": padding_mask,
            "config": self.config,
            "loss_fn": self.loss_fn,
        }
        return result

    def step(self, x_t, t, padding_mask=None, *args, **kwargs) -> dict:
        device = x_t.device
        idx = kwargs.get("idx")
        ones = torch.ones_like(t)
        t_start = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx] * ones
        t_end = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx + 1] * ones
        copied_t_start = t_start.clone().detach()
        copied_t_end = t_end.clone().detach()
        t_start: torch.Tensor = self.complete_micro_shape(copied_t_start)
        t_end: torch.Tensor = self.complete_micro_shape(copied_t_end)

        p_dx_t = self.model(x_t=x_t, t=copied_t_start, padding_mask=padding_mask, *args, **kwargs)["x"]

        hook_input = {
            "x_t": x_t,
            "t": copied_t_start,
            "padding_mask": padding_mask,
            "p_dx_t": p_dx_t,
            "sampling_condition": kwargs.get("sampling_condition"),
            "tgt_key_name": "p_dx_t",
        }

        hook_output = self.hook_manager.run_hooks(GMHookStageType.PRE_UPDATE_IN_STEP_FN, **hook_input)
        if hook_output is not None:
            vf = hook_output
        else:
            vf = p_dx_t
        return {"x": x_t + (t_end - t_start) * vf}

    @torch.no_grad()
    def sampling(
        self,
        shape,
        device,
        x_init_posterior=None,
        return_all=False,
        sampling_condition=None,
        sapmling_condition_key="sampling_condition",
        *args,
        **kwargs,
    ) -> dict:
        x_0 = self.prior_sampling(shape).to(device)
        if x_init_posterior is not None:
            x_0 = x_init_posterior * EPS + (1 - EPS) * x_0
        x_t = x_0

        masker = self.masker
        macro_shape = self.get_macro_shape(x_t)
        x_list = [x_0]

        kwargs[sapmling_condition_key] = sampling_condition
        time_steps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for idx, t in enumerate(tqdm(time_steps)):
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            kwargs["idx"] = idx
            x_t = self.step(x_t=x_t, t=t, padding_mask=no_padding_mask, *args, **kwargs)["x"]
            if return_all:
                x_list.append(x_t)
        return {"x": x_t, "x_list": x_list}

    @torch.no_grad()
    def inpainting(
        self,
        x,
        padding_mask,
        inpainting_mask,
        device,
        x_init_posterior=None,
        inpainting_mask_key="inpainting_mask",
        sapmling_condition_key="sapmling_condition",
        return_all=False,
        sampling_condition=None,
        *args,
        **kwargs,
    ) -> dict:
        x_1 = x
        shape = x_1.shape
        config = self.config
        masker = self.masker
        macro_shape = shape[: -config.ndim_micro_shape]

        x_0 = self.prior_sampling(shape).to(device)
        if x_init_posterior is not None:
            x_0 = x_init_posterior * EPS + (1 - EPS) * x_0
        x_t = x_0

        x_1 = masker.apply_mask(x_1, padding_mask)

        x_list = [x_0]

        kwargs[sapmling_condition_key] = sampling_condition
        kwargs[inpainting_mask_key] = inpainting_mask
        timesteps = self.time_scheduler.get_timestep_indices_schedule().to(device)
        for idx, t in enumerate(tqdm(timesteps)):
            kwargs["idx"] = idx
            t = torch.ones(macro_shape, device=device, dtype=torch.long) * t
            x_t = self.recover_bright_region(
                x_known=x_1,
                x_t=x_t,
                t=t,
                padding_mask=padding_mask,
                inpainting_mask=inpainting_mask,
                x_prior=x_0,
                **kwargs,
            )
            x_t = self.step(x_t=x_t, t=t, padding_mask=padding_mask, *args, **kwargs)["x"]
            x_t = masker.apply_mask(x_t, padding_mask)
            if return_all:
                x_list.append(x_t)
        x_t = masker.apply_inpainting_mask(x_1, x_t, inpainting_mask)
        return {"x": x_t, "x_list": x_list}

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(shape)

    def recover_bright_region(self, x_known, x_t, t, padding_mask, inpainting_mask, x_prior, *args, **kwargs) -> Tensor:
        device = x_t.device
        idx = kwargs.get("idx")
        t_start = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx]
        t_start = self.complete_micro_shape(t_start)
        x_1_t = t_start * x_known + (1 - t_start) * x_prior
        x_t = self.masker.apply_inpainting_mask(x_1_t, x_t, inpainting_mask)
        return x_t

    def get_posterior_mean_fn(self, vf, vf_fn=None):
        def _otfm_posterior_mean_fn(x_t, t, padding_mask):
            nonlocal vf, vf_fn
            assert vf is not None or vf_fn is not None, "Either vf or vf_fn must be provided"
            if vf is None:
                vf = vf_fn(x_t, t, padding_mask)

            t = t.view(*t.shape, *([1] * (vf.ndim - t.ndim)))
            x_1 = (1 - t) / (t + EPS) * (t * vf - x_t) + 1 / (t + EPS) * x_t
            return x_1

        return _otfm_posterior_mean_fn

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):

        def _hook_fn(**kwargs):
            nonlocal conditioner_list

            loss = kwargs.get("loss")
            x_0 = kwargs.get("x_0")
            x_t = kwargs.get("x_t")
            x_1 = kwargs.get("x_1")
            t = kwargs.get("t", None)
            padding_mask = kwargs.get("padding_mask")
            p_dx_t = kwargs.get("p_dx_t")
            loss_fn = kwargs.get("loss_fn")

            tgt_mask = padding_mask
            for conditioner in conditioner_list:
                if not conditioner.is_enabled():
                    continue
                conditioner.set_condition(
                    **{
                        **conditioner.prepare_condition_dict(
                            train=True,
                            **{
                                "tgt_mask": tgt_mask,
                                "gt_data": x_1,
                                "padding_mask": padding_mask,
                                "posterior_mean_fn": self.get_posterior_mean_fn(vf=p_dx_t, vf_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            gt_vf = (x_1 - x_0) + acc_c_score
            # Scale and compute conditioned loss

            p_vf = p_dx_t
            total_loss = loss_fn(p_vf, gt_vf, padding_mask)
            kwargs["loss"] = total_loss
            return kwargs

        return GMHook(
            name="OTFM_condition_post_compute_loss_hook",
            stage=GMHookStageType.POST_COMPUTE_LOSS,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )

    def get_condition_pre_update_in_step_fn_hook(self, conditioner_list: list[Conditioner]):
        def _hook_fn(**kwargs):
            nonlocal conditioner_list
            x_t = kwargs.get("x_t")
            t = kwargs.get("t", None)
            padding_mask = kwargs.get("padding_mask")
            p_dx_t = kwargs.get("p_dx_t")
            sampling_condition = kwargs.get("sampling_condition")

            tgt_mask = padding_mask
            for conditioner in conditioner_list:
                if not conditioner.is_enabled():
                    continue
                conditioner.set_condition(
                    **{
                        **conditioner.prepare_condition_dict(
                            train=False,
                            **{
                                "tgt_mask": tgt_mask,
                                "sampling_condition": sampling_condition,
                                "padding_mask": padding_mask,
                                "posterior_mean_fn": self.get_posterior_mean_fn(vf=p_dx_t, vf_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            vf = p_dx_t + acc_c_score
            return vf

        return GMHook(
            name="OTFM_condition_pre_update_in_step_fn_hook",
            stage=GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )
