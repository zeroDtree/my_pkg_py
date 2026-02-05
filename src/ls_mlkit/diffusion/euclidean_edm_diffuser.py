import math
from typing import Any, Callable, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.context.temp_remove import TemporaryKeyRemover
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .conditioner import Conditioner
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

        step_indices = torch.arange(n_discretization_steps, dtype=torch.float32)
        self.sigma_schedule: Tensor = (
            sigma_max ** (1 / rho)
            + step_indices / (n_discretization_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho  # max to min
        self.sigma_schedule[0] = sigma_max
        self.sigma_schedule[-1] = sigma_min
        self.sigma_schedule = self.sigma_schedule.flip(0)

        self.c_in: Tensor = 1 / torch.sqrt(self.sigma_schedule**2 + sigma_data**2)  # (n_steps,)
        self.c_noise: Tensor = 1 / 4 * torch.log(self.sigma_schedule)  # (n_steps,)

        self.c_skip: Tensor = sigma_data**2 / (self.sigma_schedule**2 + sigma_data**2)  # (n_steps,)
        self.c_out: Tensor = (
            self.sigma_schedule * sigma_data / torch.sqrt(self.sigma_schedule**2 + sigma_data**2)
        )  # (n_steps,)

    def timestep_index_to_sigma(self, timestep_index: Tensor) -> Tensor:
        """Convert discrete timesteps to sigma values.

        Args:
            discrete_t: discrete timesteps, shape=(...)

        Returns:
            sigma: noise levels, shape=(...)
        """
        timestep_index = timestep_index.clamp(0, self.n_discretization_steps - 1).long()
        return self.sigma_schedule[timestep_index].to(timestep_index.device)

    def sigma_to_timestep_index(self, sigma: Tensor) -> Tensor:
        """Convert sigma values to discrete timesteps.

        Args:
            sigma: noise levels, shape=(...)

        Returns:
            timestep_index: discrete timesteps, shape=(...)
        """
        shape = sigma.shape
        device = sigma.device
        sigma = sigma.view(-1)
        # Clamp sigma to valid range
        sigma = sigma.clamp(self.sigma_min, self.sigma_max)
        # Find nearest discrete timestep for each sigma
        sigma = sigma.unsqueeze(-1)  # (b, 1)
        schedule_expanded = self.sigma_schedule.to(device).unsqueeze(0)  # (1, b)
        distances = torch.abs(sigma - schedule_expanded)  # (b, b)
        nearest_indices = distances.argmin(dim=-1)  # (b,)
        return nearest_indices.view(shape)

    def compute_loss_weight(self, sigma: Tensor) -> Tensor:
        """Compute EDM loss weight: (sigma² + sigma_data²) / (sigma * sigma_data)².

        Args:
            sigma: noise level, shape=(...)

        Returns:
            weight: the loss weight, shape=(...)
        """
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


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
        """Initialize the EuclideanEDMDiffuser

        Args:
            config (EuclideanEDMConfig): the config of the diffuser
            time_scheduler (DiffusionTimeScheduler): the time scheduler of the diffuser
            masker (MaskerInterface): the masker of the diffuser
            model (Module): the model of the diffuser
            loss_fn (Callable[[Tensor, Tensor, Tensor], Tensor]): the loss function of the diffuser

        Returns:
            None
        """
        super().__init__(config=config, time_scheduler=time_scheduler, masker=masker)
        self.config: EuclideanEDMConfig = config
        self.model = model
        self.loss_fn = loss_fn

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        """Sample from the prior distribution (pure noise).

        For EDM, the prior is N(0, sigma_max^2 * I) since:
        x_T = x_0 + sigma_max * noise ≈ sigma_max * noise when sigma_max >> ||x_0||
        """
        return torch.randn(shape) * self.config.sigma_max

    def compute_loss(self, **batch) -> dict:
        """Compute the EDM loss.

        Args:
            **batch: batch dictionary containing:
                - gt_data: ground truth data x_0
                - padding_mask: padding mask

        Returns:
            dict: A dictionary containing the loss and other information
        """
        x_0 = batch["gt_data"]
        padding_mask = batch["padding_mask"]
        device = x_0.device

        macro_shape = self.get_macro_shape(x_0)  # (b, )
        macro_shape = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_GET_MACRO_SHAPE, tgt_key_name="macro_shape", macro_shape=macro_shape, batch=batch
        )

        t_idx = self.time_scheduler.sample_timestep_index_uniformly(macro_shape).to(device)
        t_idx = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t_idx, batch=batch
        )
        self.config = self.config.to(t_idx)
        sigma = self.config.timestep_index_to_sigma(t_idx)  # (b, )
        sigma = self.complete_micro_shape(sigma)  # (b, ..., micro_shape)

        # Forward process: add noise
        forward_result = self.forward_process(x_0, t_idx, padding_mask)
        x_t, noise = (forward_result["x_t"], forward_result["noise"])

        batch["t"] = t_idx
        batch["sigma"] = sigma
        batch["x_t"] = self.complete_micro_shape(self.config.c_in[t_idx]) * x_t

        with TemporaryKeyRemover(mapping=batch, keys=["gt_data"]):
            model_output = self.model(**batch)

        # Compute EDM loss
        p_raw = model_output["x"]
        D_yn = self._compute_denoised(x_t, p_raw, sigma)

        # EDM loss weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weight = self.config.compute_loss_weight(sigma)
        sqrt_weight = weight.sqrt()
        loss = self.loss_fn(sqrt_weight * D_yn, sqrt_weight * x_0, padding_mask)
        p_x_0 = D_yn
        print("loss:", loss)

        return {
            "loss": loss,
            "gt_data": x_0,
            "t": t_idx,
            "sigma": sigma,
            "x_t": x_t,
            "noise": noise,
            "p_raw": p_raw,
            "p_x_0": p_x_0,
            "padding_mask": padding_mask,
            "loss_fn": self.loss_fn,
            "config": self.config,
            "base_model_output": model_output,
        }

    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward process distribution q(x_t|x_0).

        For EDM, the forward process is:
        .. math::
            q(x_t|x_0) = \mathcal{N}(x_0, \sigma_t^2 I)

        Args:
            x_0 (Tensor): :math:`x_0`
            t (Tensor): discrete timesteps :math:`t`
            mask (Tensor): the mask of the sample

        Returns:
            Tuple[Tensor, Tensor]: the expectation and standard deviation of the sample
        """
        config = cast(EuclideanEDMConfig, self.config.to(t))
        sigma = config.timestep_index_to_sigma(t)
        sigma = self.complete_micro_shape(sigma)
        expectation = x_0  # EDM: mean is x_0
        standard_deviation = sigma  # EDM: std is sigma
        return expectation, standard_deviation

    def forward_process_n_step(
        self, x: Tensor, t: Tensor, next_t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Forward process from t to next_t.

        Args:
            x: the sample at timestep t
            t: the current timestep
            next_t: the next timestep
            padding_mask: the padding mask

        Returns:
            the sample at the next timestep
        """
        assert (next_t > t).all()
        assert (t >= 0).all()
        assert (next_t < self.config.n_discretization_steps).all()
        config = cast(EuclideanEDMConfig, self.config.to(t))
        sigma_t = self.complete_micro_shape(config.timestep_index_to_sigma(t))
        sigma_next = self.complete_micro_shape(config.timestep_index_to_sigma(next_t))
        # EDM forward: x_next = x + sqrt(sigma_next^2 - sigma_t^2) * noise
        noise = torch.randn_like(x)
        sigma_diff = (sigma_next**2 - sigma_t**2).clamp(min=0).sqrt()
        x_next = x + sigma_diff * noise
        return x_next

    def forward_process(
        self, x_0: Tensor, timestep_index: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        """Forward process from x_0 to x_t.

        Args:
            x_0: the sample at timestep 0
            discrete_t: discrete timesteps
            mask: the mask of the sample

        Returns:
            dict: dictionary containing:
                - x_t: noisy sample
                - noise: added noise
                - expectation: mean of the distribution
                - standard_deviation: std of the distribution
        """
        device = x_0.device
        expectation, standard_deviation = self.q_xt_x_0(x_0, timestep_index, mask)
        noise = torch.randn_like(expectation, device=device)
        x_t = expectation + standard_deviation * noise
        return {"x_t": x_t, "noise": noise, "expectation": expectation, "standard_deviation": standard_deviation}

    def _compute_denoised(self, x: Tensor, F_x: Tensor, sigma_expanded: Tensor) -> Tensor:
        """Compute denoised prediction using EDM preconditioning.

        Args:
            x: noisy input
            F_x: raw network output
            sigma_expanded: sigma value expanded to micro shape

        Returns:
            Denoised prediction D_x = c_skip * x + c_out * F_x
        """
        sigma_data_sq = self.config.sigma_data**2
        sigma_sq = sigma_expanded**2
        denom = sigma_sq + sigma_data_sq
        c_skip = sigma_data_sq / denom
        c_out = sigma_expanded * self.config.sigma_data / denom.sqrt()
        return c_skip * x + c_out * F_x

    def step(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
        r"""EDM sampling step (Euler or Heun's method).

        Args:
            x_t: the sample at timestep t
            t: the timestep (all elements must be the same)
            padding_mask: the padding mask
            use_2nd_order_correction: whether to use Heun's 2nd order correction (default: True)

        Returns:
            dict:
                - x: the sample at timestep t-1
                - E_x0_xt: the predicted original sample
        """
        assert torch.all(t == t.view(-1)[0]).item(), "All timesteps in batch must be the same for EDM step"
        config = cast(EuclideanEDMConfig, self.config.to(t))

        # Ensure t has consistent shape (batch_size,) and get next timestep
        t = t.view(-1).long()
        t_next = t - 1
        is_final_step = (t_next < 0).all()
        use_heun = not is_final_step and self.config.use_2nd_order_correction

        # Get sigma values and preconditioning coefficients with batch dimension
        sigma_cur = self.complete_micro_shape(config.timestep_index_to_sigma(t))

        if not self.config.use_ode_flow:
            episilon = self.config.S_noise * torch.randn_like(x_t)
            gamma = (
                min(self.config.S_churn / self.config.n_discretization_steps, math.sqrt(2) - 1)
                if ((self.config.S_min <= sigma_cur).all() and (sigma_cur <= self.config.S_max).all())
                else 0.0
            )
            sigma_cur_hat = sigma_cur + gamma * sigma_cur
            x_t = x_t + torch.sqrt(sigma_cur_hat**2 - sigma_cur**2) * episilon

        # p_x_0 prediction
        c_in_cur = self.complete_micro_shape(config.c_in[t])
        batch_dict = {"x_t": c_in_cur * x_t, "t": t, "sigma": sigma_cur, "padding_mask": padding_mask, **kwargs}
        F_x = self.model(**batch_dict)["x"]
        p_x_0 = self._compute_denoised(x_t, F_x, sigma_cur)

        # Run PRE_UPDATE_IN_STEP_FN hooks for conditional sampling
        hook_input = {
            "x_t": x_t,
            "t": t,
            "sigma": sigma_cur,
            "p_x_0": p_x_0,
            "p_raw": F_x,
            "padding_mask": padding_mask,
            **kwargs,
        }
        hook_output = self.hook_manager.run_hooks(
            GMHookStageType.PRE_UPDATE_IN_STEP_FN, tgt_key_name="p_x_0", **hook_input
        )
        if hook_output is not None:
            p_x_0 = hook_output

        # Final step: return denoised directly
        if is_final_step:
            return {"x": p_x_0, "E_x0_xt": p_x_0}

        # Euler step
        sigma_next = self.complete_micro_shape(config.timestep_index_to_sigma(t_next.clamp(min=0)))
        d_cur = (x_t - p_x_0) / sigma_cur.clamp(min=1e-8)
        delta_sigma = sigma_next - sigma_cur
        x_next = x_t + delta_sigma * d_cur

        # Apply Heun's 2nd order correction
        if use_heun:
            c_in_next = self.complete_micro_shape(config.c_in[t_next])
            batch_dict_next = {
                "x_t": c_in_next * x_next,  # Apply c_in scaling to match training
                "t": t_next,
                "sigma": sigma_next,
                "padding_mask": padding_mask,
                **kwargs,
            }
            F_x_next = self.model(**batch_dict_next)["x"]
            p_x_0_next = self._compute_denoised(x_next, F_x_next, sigma_next)

            hook_input = {
                "x_t": c_in_next * x_next,
                "t": t_next,
                "sigma": sigma_next,
                "p_x_0": p_x_0_next,
                "p_raw": F_x_next,
                "padding_mask": padding_mask,
                **kwargs,
            }
            hook_output = self.hook_manager.run_hooks(
                GMHookStageType.PRE_UPDATE_IN_STEP_FN, tgt_key_name="p_x_0", **hook_input
            )
            if hook_output is not None:
                p_x_0_next = hook_output
            d_prime = (x_next - p_x_0_next) / sigma_next.clamp(min=1e-8)
            x_next = x_t + 0.5 * (d_cur + d_prime) * delta_sigma

        return {"x": x_next, "E_x0_xt": p_x_0}

    def _get_previous_timestep(self, timestep: int) -> int:
        r"""Get the previous timestep for sampling.

        Args:
            timestep: current timestep

        Returns:
            the previous timestep for sampling
        """
        return timestep - 1

    def get_posterior_mean_fn(self, score: Tensor = None, score_fn: Callable = None):
        r"""Get the posterior mean function for EDM.

        For EDM, the posterior mean is:
        .. math::
            E[x_0|x_t] = D_\theta(x_t, \sigma_t)

        where D_\theta is the denoised prediction.

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
        ):
            r"""
            Args:
                x_t: shape=(..., n_nodes, 3)
                t: shape=(...), dtype=torch.long

            For EDM, the posterior mean is the denoised prediction D_\theta(x_t, \sigma_t).
            """
            nonlocal score, score_fn
            config = cast(EuclideanEDMConfig, self.config.to(t))
            sigma = self.complete_micro_shape(config.timestep_index_to_sigma(t))
            c_in = self.complete_micro_shape(config.c_in[t.long()])

            # Get model prediction and compute denoised (apply c_in scaling to match training)
            batch_dict = {"x_t": c_in * x_t, "t": t, "sigma": sigma, "padding_mask": padding_mask}
            F_x = self.model(**batch_dict)["x"]
            return self._compute_denoised(x_t, F_x, sigma)

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

    def _setup_conditioners(
        self,
        conditioner_list: list[Conditioner],
        *,
        train: bool,
        tgt_mask: Tensor,
        padding_mask: Tensor,
        p_uc_score: Tensor,
        gt_data: Tensor = None,
        sampling_condition: Tensor = None,
    ) -> None:
        """Setup conditioners with common parameters.

        Args:
            conditioner_list: list of conditioners to setup
            train: whether in training mode
            tgt_mask: target mask
            padding_mask: padding mask
            p_uc_score: unconditional predicted score
            gt_data: ground truth data (for training)
            sampling_condition: sampling condition (for inference)
        """
        posterior_mean_fn = self.get_posterior_mean_fn(score=p_uc_score, score_fn=None)

        for conditioner in conditioner_list:
            if not conditioner.is_enabled():
                continue

            if train:
                condition_dict = conditioner.prepare_condition_dict(
                    train=True,
                    tgt_mask=tgt_mask,
                    gt_data=gt_data,
                    padding_mask=padding_mask,
                    posterior_mean_fn=posterior_mean_fn,
                )
            else:
                condition_dict = conditioner.prepare_condition_dict(
                    train=False,
                    tgt_mask=tgt_mask,
                    sampling_condition=sampling_condition,
                    padding_mask=padding_mask,
                    posterior_mean_fn=posterior_mean_fn,
                )
            conditioner.set_condition(**condition_dict)

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
            sigma = kwargs["sigma"]
            padding_mask = kwargs["padding_mask"]
            loss_fn = kwargs["loss_fn"]

            # Use p_x_0 if available, otherwise compute from raw output
            p_x_0 = kwargs.get("p_x_0")
            if p_x_0 is None:
                p_x_0 = self._compute_denoised(x_t, kwargs["p_raw"], sigma)

            # Compute scores
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
            )
            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)

            # Compute conditioned loss with EDM weighting
            gt_score = gt_uc_score + acc_c_score
            gt_x_0 = x_t + sigma**2 * gt_score
            weight = self.config.compute_loss_weight(sigma)
            sqrt_weight = weight.sqrt()
            kwargs["loss"] = loss_fn(sqrt_weight * gt_x_0, sqrt_weight * p_x_0, padding_mask)
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
            sigma = kwargs["sigma"]
            padding_mask = kwargs["padding_mask"]
            sampling_condition = kwargs.get("sampling_condition")

            # Use p_x_0 if available, otherwise compute from raw output
            p_x_0 = kwargs.get("p_x_0")
            # Compute unconditional score
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
            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)

            # Compute conditioned denoised prediction: x_0 = x_t + sigma² * score
            # From: score = -(x_t - x_0) / sigma² => x_0 = x_t + sigma² * score
            sigma_squared = (sigma**2).clamp(min=1e-8)
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
