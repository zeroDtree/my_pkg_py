from typing import Any, Callable, cast

import torch
from torch import Tensor
from torch.nn import Module
from tqdm.auto import tqdm

from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.base_class.loss_mask import resolve_loss_mask
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from ..util.typing_utils import require
from .conditioner import Conditioner
from .conditioner.utils import get_accumulated_guidance
from .independent_cfm import IndependentCFMFlow, IndependentCFMFlowConfig
from .time_scheduler import FlowMatchingTimeScheduler

EPS = 1e-5


@inherit_docstrings
class RectifiedFlowConfig(IndependentCFMFlowConfig):
    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int = 2,
        n_inference_steps: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ndim_micro_shape=ndim_micro_shape,
            n_discretization_steps=n_discretization_steps,
            n_inference_steps=n_inference_steps,
            **kwargs,
        )


@inherit_docstrings
class RectifiedFlow(IndependentCFMFlow):
    def __init__(
        self,
        config: RectifiedFlowConfig,
        time_scheduler: FlowMatchingTimeScheduler,
        masker: MaskerInterface,
        model: Module,
        loss_fn: Callable,
    ) -> None:
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.config: RectifiedFlowConfig = config
        self.masker: MaskerInterface = masker
        self.model: Module = model
        self.loss_fn = loss_fn

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        return x_0 * (1 - t) + x_1 * t

    def conditional_velocity(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        return x_1 - x_0

    def recover_bright_region(
        self,
        x_known,
        x_t,
        t,
        padding_mask,
        inpainting_mask,
        x_prior,
        *args,
        **kwargs,
    ) -> Tensor:
        device = x_t.device
        idx = require(kwargs.get("idx"), "idx")
        schedule = self.time_scheduler.get_continuous_boundaries_schedule().to(device)
        t_start = schedule[int(idx)]
        t_start = self.complete_micro_shape(t_start)
        x_1_t = t_start * x_known + (1 - t_start) * x_prior
        x_t = self.masker.apply_inpainting_mask(x_1_t, x_t, inpainting_mask)
        return x_t

    def get_posterior_mean_fn(
        self,
        vf: Tensor | None,
        vf_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor] | None = None,
    ):
        def _rectified_flow_posterior_mean_fn(x_t: Tensor, t: Tensor, padding_mask: Tensor | None) -> Tensor:
            nonlocal vf, vf_fn
            assert vf is not None or vf_fn is not None, "Either vf or vf_fn must be provided"
            if vf is None:
                assert vf_fn is not None
                vf = vf_fn(x_t, t, padding_mask)

            t = t.view(*t.shape, *([1] * (vf.ndim - t.ndim)))
            return x_t + (1 - t) * vf

        return _rectified_flow_posterior_mean_fn

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):
        def _hook_fn(**kwargs: Any):
            nonlocal conditioner_list

            kwargs.get("loss")
            x_0 = require(cast(Tensor | None, kwargs.get("x_0")), "x_0")
            x_t = require(cast(Tensor | None, kwargs.get("x_t")), "x_t")
            x_1 = require(cast(Tensor | None, kwargs.get("x_1")), "x_1")
            t = require(cast(Tensor | None, kwargs.get("t")), "t")
            padding_mask = require(cast(Tensor | None, kwargs.get("padding_mask")), "padding_mask")
            p_dx_t = require(cast(Tensor | None, kwargs.get("p_dx_t")), "p_dx_t")
            loss_fn = require(cast(Callable[..., Any] | None, kwargs.get("loss_fn")), "loss_fn")

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

            acc_guidance = get_accumulated_guidance(conditioner_list, x_t, t, padding_mask)
            gt_vf = self.conditional_velocity(x_0, x_1) + acc_guidance

            p_vf = p_dx_t
            loss_mask = kwargs.get("loss_mask", padding_mask)
            total_loss = loss_fn(p_vf, gt_vf, loss_mask)
            kwargs["loss"] = total_loss
            return kwargs

        return GMHook(
            name="rectified_flow_condition_post_compute_loss_hook",
            stage=GMHookStageType.POST_COMPUTE_LOSS,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )

    def get_condition_pre_update_in_step_fn_hook(self, conditioner_list: list[Conditioner]):
        def _hook_fn(**kwargs: Any):
            nonlocal conditioner_list
            x_t = require(cast(Tensor | None, kwargs.get("x_t")), "x_t")
            t = require(cast(Tensor | None, kwargs.get("t")), "t")
            padding_mask = require(cast(Tensor | None, kwargs.get("padding_mask")), "padding_mask")
            p_dx_t = require(cast(Tensor | None, kwargs.get("p_dx_t")), "p_dx_t")
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

            acc_guidance = get_accumulated_guidance(conditioner_list, x_t, t, padding_mask)
            vf = p_dx_t + acc_guidance
            return vf

        return GMHook(
            name="rectified_flow_condition_pre_update_in_step_fn_hook",
            stage=GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )

    def compute_loss(self, **batch):
        x_1 = batch["gt_data"]
        device = x_1.device
        macro_shape: tuple[int, ...] = self.get_macro_shape(x_1)
        t = batch.get("t", torch.rand(macro_shape, device=device))
        padding_mask: Any | None = batch.get("padding_mask", None)
        copied_t = t.clone().detach()
        t: torch.Tensor = self.complete_micro_shape(t)
        x_0: torch.Tensor = self.sample_x_0(x_1)
        x_t = self.interpolate(x_0, x_1, t)
        dx_t = self.conditional_velocity(x_0, x_1)
        model_input_dict = batch
        model_input_dict.pop("gt_data")
        model_input_dict.pop("padding_mask")
        model_input_dict.pop("t", None)
        model_batch = {
            "x_t": x_t,
            "t": copied_t,
            "padding_mask": padding_mask,
            **model_input_dict,
        }
        p_dx_t = self.model(**model_batch)["x"]
        loss_mask = resolve_loss_mask(self.hook_manager, padding_mask=padding_mask, batch=model_input_dict)
        loss = self.loss_fn(p_dx_t, dx_t, loss_mask)
        result = {
            "loss": loss,
            "x_1": x_1,
            "x_t": x_t,
            "x_0": x_0,
            "t": t,
            "p_dx_t": p_dx_t,
            "padding_mask": padding_mask,
            "loss_mask": loss_mask,
            "config": self.config,
            "loss_fn": self.loss_fn,
            "batch": model_input_dict,
        }
        return result

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        device = x_t.device
        idx = require(kwargs.get("idx"), "idx")
        schedule = self.time_scheduler.get_continuous_boundaries_schedule().to(device)
        ones = torch.ones_like(t)
        t_start = schedule[int(idx)] * ones
        t_end = schedule[int(idx) + 1] * ones
        copied_t_start = t_start.clone().detach()
        copied_t_end = t_end.clone().detach()
        t_start: torch.Tensor = self.complete_micro_shape(copied_t_start)
        t_end: torch.Tensor = self.complete_micro_shape(copied_t_end)

        model_batch = {
            "x_t": x_t,
            "t": copied_t_start,
            "padding_mask": padding_mask,
            **kwargs,
        }
        p_dx_t = self.model(**model_batch)["x"]

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
            step_kwargs = {k: v for k, v in kwargs.items() if k not in ("x_t", "t", "padding_mask")}
            x_t = self.step(x_t=x_t, t=t, padding_mask=no_padding_mask, **step_kwargs)["x"]
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
            step_kwargs = {k: v for k, v in kwargs.items() if k not in ("x_t", "t", "padding_mask")}
            x_t = self.step(x_t=x_t, t=t, padding_mask=padding_mask, **step_kwargs)["x"]
            x_t = masker.apply_mask(x_t, padding_mask)
            if return_all:
                x_list.append(x_t)
        x_t = masker.apply_inpainting_mask(x_1, x_t, inpainting_mask)
        return {"x": x_t, "x_list": x_list}
