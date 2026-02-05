from typing import Any, Callable, Literal, Tuple, cast

import numpy as np
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
class EuclideanDDPMConfig(EuclideanDiffuserConfig):
    """
    Config Class for Euclidean DDPM Diffuser
    """

    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        use_probability_flow=False,
        use_clip: bool = False,
        clip_sample_range: float = 1.0,
        use_dyn_thresholding: bool = False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value: float = 1.0,
        betas=None,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            n_discretization_steps: the number of discretization steps
            ndim_micro_shape: the number of dimensions of the micro shape
            use_probability_flow: whether to use probability flow
            use_clip: whether to use clip
            clip_sample_range: the range of the clip
            use_dyn_thresholding: whether to use dynamic thresholding
            dynamic_thresholding_ratio: the ratio of the dynamic thresholding
            sample_max_value: the maximum value of the sample used in thresholding
            betas: the betas
        Returns:
            None
        """
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
        )
        self.betas: Tensor
        if betas is None:
            # Use the same beta schedule as standard DDPMScheduler
            # Linear schedule from beta_start=0.0001 to beta_end=0.02
            self.betas = torch.linspace(0.0001, 0.02, steps=self.n_discretization_steps, dtype=torch.float32)
        else:
            self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # expectation
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # std
        self.use_clip = use_clip
        self.clip_sample_range = clip_sample_range
        self.use_dyn_thresholding = use_dyn_thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value


@inherit_docstrings
class EuclideanDDPMDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        config: EuclideanDDPMConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: MaskerInterface,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted, ground_true, padding_mask)
    ):
        """Initialize the EuclideanDDPMDiffuser

        Args:
            config (EuclideanDDPMConfig): the config of the diffuser
            time_scheduler (DiffusionTimeScheduler): the time scheduler of the diffuser
            masker (MaskerInterface): the masker of the diffuser
            model (Module): the model of the diffuser
            loss_fn (Callable[[Tensor, Tensor, Tensor], Tensor]): the loss function of the diffuser

        Returns:
            None
        """
        super().__init__(config=config, time_scheduler=time_scheduler, masker=masker)
        self.config: EuclideanDDPMConfig = config
        self.model = model
        self.loss_fn = loss_fn

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(shape)

    def compute_loss(self, **batch) -> dict:
        mode: Literal["epsilon", "x_0", "score"] = batch.get("mode", "epsilon")
        x_0 = batch["gt_data"]
        padding_mask = batch["padding_mask"]
        device = x_0.device

        macro_shape = self.get_macro_shape(x_0)  # (b, )
        macro_shape = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_GET_MACRO_SHAPE, tgt_key_name="macro_shape", macro_shape=macro_shape, batch=batch
        )

        t = self.time_scheduler.sample_a_discrete_time_step_uniformly(macro_shape).to(device)  # (b, )
        t = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_SAMPLING_TIME_STEP, tgt_key_name="t", t=t, batch=batch
        )

        self.config = self.config.to(t)
        sqrt_1m_alphas_cumprod = self.complete_micro_shape(self.config.sqrt_1m_alphas_cumprod[t])
        sqrt_alphas_cumprod = self.complete_micro_shape(self.config.sqrt_alphas_cumprod[t])
        b = sqrt_1m_alphas_cumprod
        a = sqrt_alphas_cumprod

        forward_result = self.forward_process(x_0, t, padding_mask)
        x_t, noise = (forward_result["x_t"], forward_result["noise"])
        batch["t"] = t
        batch["x_t"] = x_t
        with TemporaryKeyRemover(mapping=batch, keys=["gt_data", "mode"]):
            model_output = self.model(**batch)

        # Simplified loss calculation following standard DDPM
        if mode == "epsilon":
            p_noise = model_output["x"]
            # Standard DDPM loss: MSE between predicted and actual noise
            loss = self.loss_fn(p_noise, noise, padding_mask)
            p_x_0 = (x_t - b * p_noise) / a
        elif mode == "x_0":
            p_x_0 = model_output["x"]
            # Convert to noise prediction for consistent loss calculation
            p_noise = (x_t - a * p_x_0) / b
            loss = self.loss_fn(p_noise, noise, padding_mask)
        elif mode == "score":
            raise ValueError(f"Currently not supported mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return {
            "loss": loss,
            # ======================================
            "gt_data": x_0,
            "t": t,
            "x_t": x_t,
            "noise": noise,
            "p_noise": p_noise,
            "p_x_0": p_x_0,
            "padding_mask": padding_mask,
            "a": a,
            "b": b,
            "loss_fn": self.loss_fn,
            "mode": mode,
            "config": self.config,
            # ======================================
            "base_model_output": model_output,
        }

    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward process

        .. math::

            q(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, \sqrt{1-\alpha_t} I)

        Args:
            x_0 (Tensor): :math:`x_0`
            t (Tensor): :math:`t`
            mask (Tensor): the mask of the sample

        Returns:
            Tuple[Tensor, Tensor]: the expectation and standard deviation of the sample
        """
        config = cast(EuclideanDDPMConfig, self.config.to(t))
        expectation = self.complete_micro_shape(config.sqrt_alphas_cumprod[t]) * x_0
        standard_deviation = self.complete_micro_shape(config.sqrt_1m_alphas_cumprod[t])
        return expectation, standard_deviation

    def forward_process_n_step(
        self, x: Tensor, t: Tensor, next_t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        assert (next_t > t).all()
        assert (t >= 0).all()
        assert (next_t < self.config.n_discretization_steps).all()
        config = cast(EuclideanDDPMConfig, self.config.to(t))
        a_square = config.alphas_cumprod[next_t] / config.alphas_cumprod[t]
        a = a_square**0.5
        b = (1 - a_square) ** 0.5
        a = self.complete_micro_shape(a)
        b = self.complete_micro_shape(b)
        noise = torch.randn_like(x)
        x_next = a * x + b * noise
        return x_next

    def forward_process(
        self, x_0: Tensor, discrete_t: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        device = x_0.device
        expectation, standard_deviation = self.q_xt_x_0(x_0, discrete_t, mask)
        noise = torch.randn_like(expectation, device=device)
        x_t = expectation + standard_deviation * noise
        return {"x_t": x_t, "noise": noise, "expectation": expectation, "standard_deviation": standard_deviation}

    def step(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
        r"""
        Predict the sample from the previous timestep by reversing the SDE.
        This function propagates the diffusion process from the learned model outputs.

        Based on the standard DDPM sampling formula:

        .. math::

            \hat{\mathbf{x}}_0:=\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon}_{\theta}(\mathbf{x}_t,t))

            \mathcal{N}\left( \boldsymbol{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\boldsymbol{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{\boldsymbol{x}}_0}{1-\bar{\alpha}_t}}_{\mu_q(\boldsymbol{x}_t, \hat{\boldsymbol{x}}_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\Sigma_q(t)} \right)

        Args:
            x_t (Tensor): the sample at timestep t
            t (Tensor): the timestep
            padding_mask (Tensor): the padding mask

        Returns:
            dict:
                "x": the sample at timestep t-1
                "E_x0_xt": the predicted original sample
        """
        mode: Literal["epsilon", "x_0", "score"] = kwargs.get("mode", "epsilon")
        assert torch.all(t == t.view(-1)[0]).item()
        config = cast(EuclideanDDPMConfig, self.config.to(t))

        # Convert to scalar timestep for indexing
        t_scalar = t.view(-1)[0].long()

        # Get model prediction
        model_output = self.model(x_t, t.long(), padding_mask, *args, **kwargs)

        if mode == "epsilon":
            model_pred = model_output["x"]
            hook_input = {
                "x_t": x_t,
                "t": t,
                "p_noise": model_pred,
                "padding_mask": padding_mask,
                "config": self.config,
                "sampling_condition": kwargs.get("sampling_condition"),
                "b": self.complete_micro_shape(self.config.sqrt_1m_alphas_cumprod[t]),
            }
            hook_output = self.hook_manager.run_hooks(
                GMHookStageType.PRE_UPDATE_IN_STEP_FN, tgt_key_name="p_noise", **hook_input
            )
            if hook_output is not None:
                model_pred = hook_output
        elif mode == "x_0":
            raise ValueError(f"Currently not supported mode: {mode}")
            model_pred = model_output["x"]
        elif mode == "score":
            raise ValueError(f"Currently not supported mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Calculate previous timestep (handle both standard and custom timestep schedules)
        prev_t = self._get_previous_timestep(t_scalar)

        # Get alpha values
        alpha_prod_t = config.alphas_cumprod[t_scalar]
        alpha_prod_t_prev = config.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0).to(t.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute predicted original sample from predicted noise
        pred_original_sample: Tensor = None
        if mode == "epsilon":
            pred_original_sample = (x_t - beta_prod_t**0.5 * model_pred) / alpha_prod_t**0.5
        elif mode == "x_0":
            raise ValueError(f"Currently not supported mode: {mode}")
            pred_original_sample = model_pred

        # Clip predicted x_0 (following standard DDPM implementation)
        # 3. Clip or threshold "predicted x_0"
        if self.config.use_dyn_thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.use_clip:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )
        # Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://huggingface.co/papers/2006.11239
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t

        # Add noise (variance) - following standard DDPM variance calculation
        variance = 0
        if t_scalar > 0:
            # Standard DDPM variance: β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
            variance_value = self._get_variance(t_scalar, alpha_prod_t, alpha_prod_t_prev, current_beta_t)
            variance_noise = torch.randn_like(x_t)
            variance = (variance_value**0.5) * variance_noise
            pred_prev_sample = pred_prev_sample + variance

        return {
            "x": pred_prev_sample,
            "E_x0_xt": pred_original_sample,
        }

    def _get_previous_timestep(self, timestep: int) -> int:
        r"""Get the previous timestep for sampling.

        Args:
            timestep (int): timestep

        Returns:
            int: the previous timestep for sampling
        """
        return timestep - 1

    def _get_variance(self, t: int, alpha_prod_t: Tensor, alpha_prod_t_prev: Tensor, current_beta_t: Tensor) -> Tensor:
        r"""Calculate variance for timestep t following standard DDPM formula. For t > 0, compute predicted variance βt (see formula (6) and (7) from https://huggingface.co/papers/2006.11239)

        .. math::

            \sigma^2 = (\frac{1 - \bar{\alpha}_{pre}}{1 - \bar{\alpha}_{t}}) \cdot ( 1- \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{pre}})

        Args:
            t (int): timestep
            alpha_prod_t (Tensor): :math:`\bar{\alpha}_t`
            alpha_prod_t_prev (Tensor): :math:`\bar{\alpha}_{t-1}`
            current_beta_t (Tensor): :math:`\beta_t`

        Returns:
            Tensor: the variance for timestep t
        """
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # Clamp variance to ensure numerical stability
        variance = torch.clamp(variance, min=1e-20)
        return variance

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

    def get_posterior_mean_fn(self, score: Tensor = None, score_fn: Callable = None):
        r"""Get the posterior mean function

        Args:
            score (Tensor, optional): the score of the sample
            score_fn (Callable, optional): the function to compute score

        Returns:
            Callable: the posterior mean function
        """

        def _ddpm_posterior_mean_fn(
            x_t: Tensor,
            t: Tensor,
            padding_mask: Tensor,
        ):
            r"""
            Args:
                x_t: shape=(..., n_nodes, 3)
                t: shape=(...), dtype=torch.long

            For the case of DDPM sampling, the posterior mean is given by

            .. math::

                E[x_0|x_t] = \frac{1}{\sqrt{\bar{\alpha}(t)}}(x_t + (1 - \bar{\alpha}(t))\nabla_{x_t}\log p_t(x_t))

            """
            nonlocal score, score_fn
            assert score is not None or score_fn is not None, "either score or score_fn must be provided"
            t = t.view(*t.shape, *([1] * (x_t.ndim - t.ndim)))
            if score is None:
                score = score_fn(x_t, t, padding_mask)
            config = cast(EuclideanDDPMConfig, self.config.to(t))
            alpha_bar_t = config.alphas_cumprod[t]  # macro_shape
            alpha_bar_t.view(*alpha_bar_t.shape, *([1] * config.ndim_micro_shape))
            x_0 = (x_t + (1 - alpha_bar_t) * score) / torch.sqrt(alpha_bar_t)
            return x_0

        return _ddpm_posterior_mean_fn

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):

        def _hook_fn(**kwargs):
            nonlocal conditioner_list
            x_0 = kwargs.get("gt_data")
            x_t = kwargs.get("x_t")
            t = kwargs.get("t", None)
            noise = kwargs.get("noise", None)
            p_noise = kwargs.get("p_noise")
            padding_mask = kwargs.get("padding_mask")
            b = kwargs.get("b")
            loss_fn = kwargs.get("loss_fn")

            p_uc_score = -p_noise / b
            gt_uc_score = -noise / b

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
                                "gt_data": x_0,
                                "padding_mask": padding_mask,
                                "posterior_mean_fn": self.get_posterior_mean_fn(score=p_uc_score, score_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            gt_score = gt_uc_score + acc_c_score

            # Scale and compute conditioned loss
            p_uc_score = b * p_uc_score
            gt_score = b * gt_score
            new_loss = loss_fn(p_uc_score, gt_score, padding_mask)
            kwargs["loss"] = new_loss
            return kwargs

        return GMHook(
            name="DDPM_condition_post_compute_loss_hook",
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
            p_noise = kwargs.get("p_noise")
            padding_mask = kwargs.get("padding_mask")
            b = kwargs.get("b")
            sampling_condition = kwargs.get("sampling_condition")
            p_uc_score = -p_noise / b

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
                                "posterior_mean_fn": self.get_posterior_mean_fn(score=p_uc_score, score_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            # Scale and compute conditioned loss
            p_epsilon = -b * (p_uc_score + acc_c_score)
            return p_epsilon

        return GMHook(
            name="DDPM_condition_pre_update_in_step_fn_hook",
            stage=GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )
