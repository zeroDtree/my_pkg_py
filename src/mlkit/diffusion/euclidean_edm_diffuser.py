import math
from typing import Any, Callable, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.base_class.loss_mask import resolve_loss_mask
from ..util.context.temp_remove import TemporaryKeyRemover
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .conditioner import Conditioner, LGDConditioner
from .conditioner.utils import get_accumulated_conditional_score
from .euclidean_diffuser import EuclideanDiffuser, EuclideanDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class EuclideanEDMConfig(EuclideanDiffuserConfig):
    """
    Config Class for Euclidean EDM Diffuser
    """

    def __init__(
        self,
        n_discretization_steps: int = 200,
        ndim_micro_shape: int = 2,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        use_2nd_order_correction: bool = True,
        use_ode_flow: bool = False,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        use_clip: bool = False,
        clip_sample_range: float = 1.0,
        use_dyn_thresholding: bool = False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value: float = 1.0,
        sigma_multiply_by_sigma_data: bool = False,
        do_forward_process: bool = True,
        do_edm_combine: bool = True,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            n_discretization_steps: the number of discretization steps
            ndim_micro_shape: the number of dimensions of the micro shape
            P_mean: mean of the log-normal distribution for sampling sigma during training
            P_std: standard deviation of the log-normal distribution for sampling sigma during training
            sigma_data: expected standard deviation of the training data
            sigma_min: minimum supported noise level
            sigma_max: maximum supported noise level
            rho: time step exponent for sampling schedule
            do_forward_process: If True (BioDiff default), sample sigma and run
                ``forward_process``. If False (RFD3 path), use batch ``x_t`` / ``t``.
            do_edm_combine: If True (BioDiff default), pass ``c_in·x_t`` to the model
                and form ``p_x_0 = c_skip·x_t + c_out·F``. If False (RFD3 path), pass
                physical ``x_t`` and treat ``model["x"]`` as denoiser ``D`` / ``p_x_0``.
        Returns:
            None
        """
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
        )
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.use_ode_flow = use_ode_flow
        self.use_2nd_order_correction = use_2nd_order_correction
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        self.use_clip = use_clip
        self.clip_sample_range = clip_sample_range
        self.use_dyn_thresholding = use_dyn_thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value

        self.sigma_multiply_by_sigma_data = sigma_multiply_by_sigma_data
        self.do_forward_process = do_forward_process
        self.do_edm_combine = do_edm_combine

        step_indices = torch.arange(n_discretization_steps + 1, dtype=torch.float32)
        self.sigma_schedule: Tensor = (
            sigma_min ** (1 / rho)
            + (step_indices - 1) / (n_discretization_steps - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))
        ) ** rho
        self.sigma_schedule[0] = 0.0

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma: Tensor) -> Tensor:
        return 1 / 4 * torch.log(sigma)

    def c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)

    @staticmethod
    def _expand_coeff(coeff: Tensor, ref: Tensor) -> Tensor:
        """Broadcast per-batch EDM coefficients to match ``ref`` rank."""
        while coeff.dim() < ref.dim():
            coeff = coeff.unsqueeze(-1)
        return coeff

    def scale_r_pred(self, F_x: Tensor, x_scaled: Tensor, sigma: Tensor) -> Tensor:
        """Map network output to unit-data prediction under EDM preconditioning.

        Expects ``x_scaled = c_in(sigma) * x_t``. Returns ``detach(x_pred / sigma_data)``.
        Used by recycle self-conditioning to seed the next pass.

        Args:
            F_x: raw network output, same shape as ``x_scaled``.
            x_scaled: EDM-preconditioned noisy input ``c_in(sigma) * x_t``.
            sigma: noise level with batch dim matching ``F_x`` leading dim.

        Returns:
            Detached ``x_pred / sigma_data``.
        """
        c_in = self._expand_coeff(self.c_in(sigma), x_scaled)
        c_skip = self._expand_coeff(self.c_skip(sigma), x_scaled)
        c_out = self._expand_coeff(self.c_out(sigma), x_scaled)
        x_t = x_scaled / c_in
        x_pred = c_skip * x_t + c_out * F_x
        return (x_pred / self.sigma_data).detach()

    def sigma(self, t: Tensor, is_continuous_time: bool = True) -> Tensor:
        if is_continuous_time:
            return t
        else:
            return self.timestep_index_to_sigma(t)

    def timestep_index_to_sigma(self, timestep_index: Tensor) -> Tensor:
        """Convert discrete timesteps to sigma values.

        Args:
            discrete_t: discrete timesteps, shape=(...)

        Returns:
            sigma: noise levels, shape=(...)
        """
        timestep_index = timestep_index.clamp(1, self.n_discretization_steps).long()
        return self.sigma_schedule[timestep_index].to(timestep_index.device)

    def compute_loss_weight(self, sigma: Tensor) -> Tensor:
        """Compute EDM loss weight: (sigma² + sigma_data²) / (sigma * sigma_data)².

        Args:
            sigma: noise level, shape=(...)

        Returns:
            weight: the loss weight, shape=(...)
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def sampling_timestep_for_training(self, macro_shape: tuple):
        rnd_normal = torch.randn(macro_shape)
        t = (self.P_mean + self.P_std * rnd_normal).exp()
        if self.sigma_multiply_by_sigma_data:
            t = t * self.sigma_data
        return t


@inherit_docstrings
class EuclideanEDMDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        config: EuclideanEDMConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: MaskerInterface,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted, ground_true, padding_mask)
    ):
        super().__init__(config=config, time_scheduler=time_scheduler, masker=masker)
        self.config: EuclideanEDMConfig = config
        self.model = model
        self.loss_fn = loss_fn

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(shape) * self.config.sigma_max

    def _resolve_xt_sigma_for_training(
        self,
        batch: dict,
        x_0: Tensor,
        padding_mask: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Resolve physical ``x_t``, ``t``, ``sigma``, and optional ``noise`` for training.

        When ``do_forward_process`` is True, sample ``t`` and run ``forward_process``.
        When False, require batch ``x_t`` / ``t`` (physical noisy coords; continuous sigma).
        """
        if self.config.do_forward_process:
            macro_shape = self.get_macro_shape(x_0)  # (b, )
            macro_shape = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_GET_MACRO_SHAPE,
                tgt_key_name="macro_shape",
                macro_shape=macro_shape,
                batch=batch,
            )
            macro_shape = cast(tuple[int, ...], macro_shape)
            t = self.config.sampling_timestep_for_training(macro_shape=macro_shape).to(device)
            t = self.hook_manager.run_hooks(
                stage=GMHookStageType.POST_SAMPLING_TIME_STEP,
                tgt_key_name="t",
                t=t,
                batch=batch,
            )
            t = cast(Tensor, t)
            t = self.complete_micro_shape(t)

            forward_result = self.forward_process(
                x_0, torch.zeros_like(t), t, padding_mask, is_continuous_time=True
            )
            x_t = forward_result["x_t"]
            noise = forward_result["noise"]
            sigma = forward_result["sigma_diff"]
            return x_t, t, sigma, noise

        if "x_t" not in batch or "t" not in batch:
            raise KeyError(
                "When do_forward_process=False, batch must provide physical 'x_t' and 't'."
            )
        t = cast(Tensor, batch["t"]).to(device)
        t = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_SAMPLING_TIME_STEP,
            tgt_key_name="t",
            t=t,
            batch=batch,
        )
        t = cast(Tensor, t)
        t = self.complete_micro_shape(t)
        x_t = cast(Tensor, batch["x_t"]).to(device)
        sigma = self.config.sigma(t, is_continuous_time=True)
        noise = cast(Tensor | None, batch.get("noise"))
        return x_t, t, sigma, noise

    def _prepare_model_x_t(self, x_t: Tensor, sigma: Tensor) -> tuple[Tensor, dict]:
        """Prepare model input ``x_t`` and ``gm_kwargs`` from physical ``x_t``.

        When ``do_edm_combine`` is True, scale by ``c_in`` and pass it in ``gm_kwargs``.
        When False, pass physical ``x_t`` with empty ``gm_kwargs``.
        """
        if self.config.do_edm_combine:
            c_in = self.config.c_in(sigma)
            return c_in * x_t, {"c_in": c_in}
        return x_t, {}

    def _resolve_p_x_0(self, x_t: Tensor, model_x: Tensor, sigma: Tensor) -> Tensor:
        """Resolve denoised prediction ``p_x_0`` from model output.

        When ``do_edm_combine`` is True, apply EDM combine ``c_skip·x_t + c_out·F``.
        When False, treat ``model_x`` as denoiser ``D`` / ``p_x_0``.
        """
        if self.config.do_edm_combine:
            return self._compute_denoised(x_t, model_x, sigma)
        return model_x

    def compute_loss(self, **batch) -> dict:
        """Compute the EDM loss.

        Args:
            **batch: batch dictionary containing:
                - gt_data: ground truth data x_0
                - padding_mask: padding mask
                - x_t / t: required when ``do_forward_process`` is False

        Returns:
            dict: A dictionary containing the loss and other information
        """
        x_0 = batch["gt_data"]
        padding_mask = batch["padding_mask"]
        device = x_0.device

        x_t, t, sigma, noise = self._resolve_xt_sigma_for_training(
            batch=batch, x_0=x_0, padding_mask=padding_mask, device=device
        )
        model_x_t, gm_kwargs = self._prepare_model_x_t(x_t, sigma)
        batch["t"] = t
        batch["x_t"] = model_x_t
        batch["gm_kwargs"] = gm_kwargs

        with TemporaryKeyRemover(mapping=batch, keys=["gt_data"]):
            model_output = self.model(**batch)

        # Compute EDM loss (always data-space ||D - x_0||^2 with lambda(sigma))
        p_raw = model_output["x"]
        p_x_0 = self._resolve_p_x_0(x_t, p_raw, sigma)

        # EDM loss weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weight = self.config.compute_loss_weight(sigma)
        sqrt_weight = weight.sqrt()
        loss_mask = resolve_loss_mask(self.hook_manager, padding_mask=padding_mask, batch=batch)
        loss = self.loss_fn(sqrt_weight * p_x_0, sqrt_weight * x_0, loss_mask)

        return {
            "loss": loss,
            "gt_data": x_0,
            "t": t,
            "sigma": sigma,
            "x_t": x_t,
            "noise": noise,
            "p_raw": p_raw,
            "p_x_0": p_x_0,
            "padding_mask": padding_mask,
            "loss_mask": loss_mask,
            "loss_fn": self.loss_fn,
            "config": self.config,
            "base_model_output": model_output,
            "batch": batch,
        }

    def forward_process(
        self,
        x_0: Tensor,
        t_a: Tensor,
        t_b: Tensor,
        mask: Tensor,
        is_continuous_time: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        assert (t_b >= t_a).all()
        sigma_a = self.config.sigma(t_a, is_continuous_time)
        sigma_b = self.config.sigma(t_b, is_continuous_time)
        sigma_diff = (sigma_b**2 - sigma_a**2).clamp(min=0).sqrt()
        noise = torch.randn_like(x_0)
        x_t = x_0 + sigma_diff * noise
        return {"x_t": x_t, "noise": noise, "sigma_diff": sigma_diff}

    def _compute_denoised(self, x: Tensor, F_x: Tensor, sigma_expanded: Tensor) -> Tensor:
        """Compute denoised prediction using EDM preconditioning.

        Args:
            x: noisy input
            F_x: raw network output
            sigma_expanded: sigma value expanded to micro shape

        Returns:
            Denoised prediction D_x = c_skip * x + c_out * F_x
        """

        return self.config.c_skip(sigma_expanded) * x + self.config.c_out(sigma_expanded) * F_x

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        r"""EDM sampling step (Euler or Heun's method).

        Args:
            x_t: the sample at timestep t
            t: the timestep (all elements must be the same)
            padding_mask: the padding mask

        Returns:
            dict:
                - x: the sample at timestep t-1
                - E_x0_xt: the predicted original sample
                - base_model_output: full model forward output (optional, for auxiliary heads)
        """
        assert torch.all(t == t.view(-1)[0]).item(), "All timesteps in batch must be the same for EDM step"
        assert t.ndim == x_t.ndim, "Timestep and sample must have the same number of dimensions"
        config = cast(EuclideanEDMConfig, self.config.to(t))
        t = t.long()
        t_next = t - 1
        is_final_step = (t_next == 0).all()
        use_heun = not is_final_step and self.config.use_2nd_order_correction

        # Get sigma values and preconditioning coefficients with batch dimension
        sigma_cur = config.sigma(t, is_continuous_time=False)

        if not self.config.use_ode_flow:
            epsilon = self.config.S_noise * torch.randn_like(x_t)
            gamma = (
                min(
                    self.config.S_churn / self.config.n_discretization_steps,
                    math.sqrt(2) - 1,
                )
                if ((self.config.S_min <= sigma_cur).all() and (sigma_cur <= self.config.S_max).all())
                else 0.0
            )
            sigma_cur_hat = sigma_cur + gamma * sigma_cur
            x_t = x_t + torch.sqrt(sigma_cur_hat**2 - sigma_cur**2) * epsilon

        # p_x_0 prediction
        model_x_t, gm_kwargs = self._prepare_model_x_t(x_t, sigma_cur)
        batch_dict = {
            "x_t": model_x_t,
            "t": sigma_cur,
            "padding_mask": padding_mask,
            **kwargs,
            "gm_kwargs": gm_kwargs,
        }
        model_output = self.model(**batch_dict)
        F_x = model_output["x"]
        base_model_output = model_output
        p_x_0 = self._resolve_p_x_0(x_t, F_x, sigma_cur)

        # Clip predicted x_0 (following standard DDPM implementation)
        # 3. Clip or threshold "predicted x_0"
        if self.config.use_dyn_thresholding:
            p_x_0 = self._threshold_sample(p_x_0)
        elif self.config.use_clip:
            p_x_0 = p_x_0.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        # Run PRE_UPDATE_IN_STEP_FN hooks for conditional sampling
        hook_input = {
            "x_t": x_t,
            "t": sigma_cur,
            "p_x_0": p_x_0,
            "p_raw": F_x,
            "padding_mask": padding_mask,
            **kwargs,
        }
        hook_output = self.hook_manager.run_hooks(
            GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            tgt_key_name="p_x_0",
            **hook_input,
        )
        if hook_output is not None:
            p_x_0 = hook_output

        # Final step: return denoised directly
        if is_final_step:
            return {"x": p_x_0, "E_x0_xt": p_x_0, "base_model_output": base_model_output}

        # Euler step
        sigma_next = config.sigma(t_next, is_continuous_time=False)
        d_cur = (x_t - p_x_0) / sigma_cur.clamp(min=1e-8)
        delta_sigma = sigma_next - sigma_cur
        x_next = x_t + delta_sigma * d_cur

        # Apply Heun's 2nd order correction
        if use_heun:
            model_x_next, gm_kwargs_next = self._prepare_model_x_t(x_next, sigma_next)
            batch_dict_next = {
                "x_t": model_x_next,
                "t": sigma_next,
                "padding_mask": padding_mask,
                **kwargs,
                "gm_kwargs": gm_kwargs_next,
            }
            model_output_next = self.model(**batch_dict_next)
            F_x_next = model_output_next["x"]
            base_model_output = model_output_next
            p_x_0_next = self._resolve_p_x_0(x_next, F_x_next, sigma_next)

            hook_input = {
                "x_t": x_next,
                "t": sigma_next,
                "p_x_0": p_x_0_next,
                "p_raw": F_x_next,
                "padding_mask": padding_mask,
                **kwargs,
            }
            hook_output = self.hook_manager.run_hooks(
                GMHookStageType.PRE_UPDATE_IN_STEP_FN,
                tgt_key_name="p_x_0",
                **hook_input,
            )
            if hook_output is not None:
                p_x_0_next = hook_output
            d_prime = (x_next - p_x_0_next) / sigma_next.clamp(min=1e-8)
            x_next = x_t + 0.5 * (d_cur + d_prime) * delta_sigma

        return {"x": x_next, "E_x0_xt": p_x_0, "base_model_output": base_model_output}

    def get_posterior_mean_fn(
        self, score: Optional[Tensor] = None, score_fn: Optional[Callable] = None, batch: Optional[dict] = None
    ):
        r"""Get the posterior mean function for EDM.

        For EDM, the posterior mean is:

        $$
        E[x_0|x_t] = D_\theta(x_t, \sigma_t)
        $$

        where $D_\theta$ is the denoised prediction.

        Args:
            score (Tensor, optional): the score of the sample
            score_fn (Callable, optional): the function to compute score

        Returns:
            Callable: the posterior mean function
        """

        def _edm_posterior_mean_fn(
            x_t: Tensor,
            t: Tensor,
            padding_mask: Tensor,
            is_continuous_time: bool = True,
        ):
            r"""
            Args:
                x_t: shape=(..., n_nodes, 3)
                t: shape=(...), dtype=torch.long

            For EDM, the posterior mean is the denoised prediction D_\theta(x_t, \sigma_t).
            """
            # TODO: get x0 by score function
            nonlocal score, score_fn
            sigma = self.config.sigma(t, is_continuous_time=True)
            model_x_t, gm_kwargs = self._prepare_model_x_t(x_t, sigma)
            batch_dict = {
                "x_t": model_x_t,
                "t": t,
                "sigma": sigma,
                "padding_mask": padding_mask,
                "gm_kwargs": gm_kwargs,
            }
            if batch is not None and "features" in batch:
                batch_dict["features"] = batch["features"]
            F_x = self.model(**batch_dict)["x"]
            return self._resolve_p_x_0(x_t, F_x, sigma)

        return _edm_posterior_mean_fn

    def _compute_edm_score(self, x_t: Tensor, x_0: Tensor, sigma: Tensor) -> Tensor:
        """Compute EDM score function: -(x_t - x_0) / sigma².

        Args:
            x_t: noisy sample at time t
            x_0: clean sample (predicted or ground truth)
            sigma: noise level

        Returns:
            score: the score function value
        """
        sigma_squared = (sigma**2).clamp(min=1e-8)
        return -(x_t - x_0) / sigma_squared

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):
        """Get hook for conditioning after loss computation (training).

        This hook modifies the loss to include conditional guidance during training.
        It computes the conditional score and updates the loss accordingly.

        Args:
            conditioner_list: list of conditioners

        Returns:
            GMHook: the hook for POST_COMPUTE_LOSS stage
        """

        def _hook_fn(**kwargs):
            x_0 = kwargs["gt_data"]
            x_t = kwargs["x_t"]
            t = kwargs["t"]
            padding_mask = kwargs["padding_mask"]
            loss_fn = kwargs["loss_fn"]
            batch = kwargs.get("batch")

            # Use p_x_0 if available, otherwise compute from raw output
            p_x_0 = kwargs.get("p_x_0")

            # Compute scores
            sigma = self.config.sigma(t, is_continuous_time=True)
            p_x_0 = cast(Tensor, p_x_0)
            p_uc_score = self._compute_edm_score(x_t, p_x_0, sigma)
            gt_uc_score = self._compute_edm_score(x_t, x_0, sigma)

            # Setup conditioners and get accumulated conditional score
            self._setup_conditioners(
                conditioner_list,
                train=True,
                tgt_mask=padding_mask,
                padding_mask=padding_mask,
                p_uc_score=p_uc_score,
                gt_data=x_0,
                batch=batch,
            )
            acc_c_score = get_accumulated_conditional_score(
                conditioner_list, x_t, t, padding_mask, is_continuous_time=True
            )

            # Collect per-conditioner metrics for monitoring
            conditioner_metrics: dict[str, float] = {
                "LGD-acc_cond_score_norm": float(acc_c_score.detach().norm()),
            }
            for cond in conditioner_list:
                if cond.is_enabled() and isinstance(cond, LGDConditioner):
                    for k, v in cond.last_step_metrics.items():
                        conditioner_metrics[f"LGD-{k}"] = v
            kwargs["conditioner_metrics"] = conditioner_metrics

            # Compute conditioned loss with EDM weighting
            gt_score = gt_uc_score + acc_c_score
            gt_x_0 = x_t + sigma**2 * gt_score
            weight = self.config.compute_loss_weight(sigma)
            sqrt_weight = weight.sqrt()
            loss_mask = kwargs.get("loss_mask", padding_mask)
            kwargs["loss"] = loss_fn(sqrt_weight * gt_x_0, sqrt_weight * p_x_0, loss_mask)
            return kwargs

        return GMHook(
            name="EDM_condition_post_compute_loss_hook",
            stage=GMHookStageType.POST_COMPUTE_LOSS,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )

    def get_condition_pre_update_in_step_fn_hook(self, conditioner_list: list[Conditioner]):
        """Get hook for conditioning before update in step function (sampling).

        This hook applies conditional guidance during sampling by modifying
        the predicted denoised sample based on the conditional score.

        Args:
            conditioner_list: list of conditioners

        Returns:
            GMHook: the hook for PRE_UPDATE_IN_STEP_FN stage
        """

        def _hook_fn(**kwargs):
            x_t = kwargs["x_t"]
            t = kwargs["t"]
            padding_mask = kwargs["padding_mask"]
            sampling_condition = kwargs.get("sampling_condition")

            # Use p_x_0 if available, otherwise compute from raw output
            p_x_0 = kwargs.get("p_x_0")
            # Compute unconditional score
            sigma = self.config.sigma(t, is_continuous_time=True)
            p_x_0 = cast(Tensor, p_x_0)
            p_uc_score = self._compute_edm_score(x_t, p_x_0, sigma)

            # Setup conditioners and get accumulated conditional score
            self._setup_conditioners(
                conditioner_list,
                train=False,
                tgt_mask=padding_mask,
                padding_mask=padding_mask,
                p_uc_score=p_uc_score,
                sampling_condition=sampling_condition,
            )
            acc_c_score = get_accumulated_conditional_score(
                conditioner_list, x_t, t, padding_mask, is_continuous_time=True
            )

            # Compute conditioned denoised prediction: x_0 = x_t + sigma² * score
            # From: score = -(x_t - x_0) / sigma² => x_0 = x_t + sigma² * score
            sigma_squared = sigma**2
            p_c_x_0 = x_t + sigma_squared * (p_uc_score + acc_c_score)

            # Return p_c_x_0 directly (hook manager expects target value when tgt_key_name is set)
            return p_c_x_0

        return GMHook(
            name="EDM_condition_pre_update_in_step_fn_hook",
            stage=GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )
