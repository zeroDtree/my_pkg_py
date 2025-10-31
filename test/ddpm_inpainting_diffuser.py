from typing import Any, Callable, Literal, Tuple, cast

import numpy as np
import torch
from torch import Tensor

from ls_mlkit.my_diffuser.time_scheduler import TimeScheduler
from ls_mlkit.my_utils.decorators import inherit_docstrings
from ls_mlkit.my_utils.mask.masker_interface import MaskerInterface

from ls_mlkit.my_diffuser.conditioner import Conditioner
from ls_mlkit.my_diffuser.euclidean_ddpm_diffuser import EuclideanDDPMConfig, EuclideanDDPMDiffuser
from ls_mlkit.my_diffuser.model_interface import Model4DiffuserInterface
from ls_mlkit.my_utils.decorators import inherit_docstrings


@inherit_docstrings
class EuclideanDDPMDiffuser(EuclideanDDPMDiffuser):
    def __init__(
        self,
        config: EuclideanDDPMConfig,
        time_scheduler: TimeScheduler,
        masker: MaskerInterface,
        conditioner_list: list[Conditioner],
        model: Model4DiffuserInterface,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted, ground_true, padding_mask)
    ):
        """Initialize the EuclideanDDPMDiffuser

        Args:
            config (EuclideanDDPMConfig): the config of the diffuser
            time_scheduler (TimeScheduler): the time scheduler of the diffuser
            masker (MaskerInterface): the masker of the diffuser
            conditioner_list (list[Conditioner]): the list of conditioners of the diffuser
            model (Model4DiffuserInterface): the model of the diffuser
            loss_fn (Callable[[Tensor, Tensor, Tensor], Tensor]): the loss function of the diffuser

        Returns:
            None
        """
        super().__init__(config=config, time_scheduler=time_scheduler, masker=masker, conditioner_list=conditioner_list)
        self.config: EuclideanDDPMConfig = config
        self.model = model
        self.loss_fn = loss_fn

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        return torch.randn(shape)

    def compute_loss(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> dict:
        mode: Literal["epsilon", "x_0", "score"] = batch.get("mode", "epsilon")
        batch = self.model.prepare_batch_data_for_input(batch)
        assert isinstance(batch, dict), "batch must be a dictionary"
        x_0 = batch["x_0"]
        padding_mask = batch["padding_mask"]
        device = x_0.device
        macro_shape = self.get_macro_shape(x_0)

        t = batch.get("t", None)
        if t is None:
            t = self.time_scheduler.sample_a_discrete_time_step_uniformly(macro_shape).to(device)
        self.config = self.config.to(t)
        sqrt_1m_alphas_cumprod = self.complete_micro_shape(self.config.sqrt_1m_alphas_cumprod[t])
        sqrt_alphas_cumprod = self.complete_micro_shape(self.config.sqrt_alphas_cumprod[t])
        b = sqrt_1m_alphas_cumprod
        a = sqrt_alphas_cumprod

        forward_result = self.forward_process(x_0, t, padding_mask)
        x_t, noise = (forward_result["x_t"], forward_result["noise"])

        model_input_dict = batch
        model_input_dict.pop("x_0")
        model_input_dict.pop("padding_mask")
        model_input_dict.pop("t", None)
        model_output = self.model(x_t, t, padding_mask, **model_input_dict)

        # Simplified loss calculation following standard DDPM
        if mode == "epsilon":
            predicted_noise = model_output["x"]
            # Standard DDPM loss: MSE between predicted and actual noise
            loss = self.loss_fn(predicted_noise, noise, padding_mask)
        elif mode == "x_0":
            predicted_x0 = model_output["x"]
            # Convert to noise prediction for consistent loss calculation
            predicted_noise = (x_t - a * predicted_x0) / b
            loss = self.loss_fn(predicted_noise, noise, padding_mask)
        elif mode == "score":
            raise ValueError(f"Currently not supported mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Handle conditioners if any (for advanced use cases)
        if len(self.conditioner_list) > 0:
            # Original complex logic for conditioners
            p_uc_score = -predicted_noise / b
            gt_uc_score = -noise / b

            tgt_mask = padding_mask
            for conditioner in self.conditioner_list:
                conditioner.set_condition(
                    **{
                        **conditioner.prepare_condition_dict(
                            train=True,
                            **{
                                "tgt_mask": tgt_mask,
                                "x_0": x_0,
                                "padding_mask": padding_mask,
                                "posterior_mean_fn": self.get_posterior_mean_fn(score=p_uc_score, score_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = self.get_accumulated_conditional_score(x_t, t, padding_mask)
            gt_score = gt_uc_score + acc_c_score

            # Scale and compute conditioned loss
            p_uc_score = b * p_uc_score
            gt_score = b * gt_score
            loss = self.loss_fn(p_uc_score, gt_score, padding_mask)
        return {"loss": loss, "model_output": model_output}

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

    def forward_process_one_step(self, x: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        config = cast(EuclideanDDPMConfig, self.config.to(t))
        beta_t = config.betas[t]  # (macro_shape)
        a = (1 - beta_t) ** 0.5
        b = beta_t**0.5
        a = self.complete_micro_shape(a)
        b = self.complete_micro_shape(b)
        noise = torch.randn_like(x)
        x_next = a * x + b * noise
        x_next = self.masker.apply_mask(x_next, padding_mask)
        return x_next

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
        x_next = self.masker.apply_mask(x_next, padding_mask)
        return x_next

    def forward_process(
        self, x_0: Tensor, discrete_t: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        device = x_0.device
        expectation, standard_deviation = self.q_xt_x_0(x_0, discrete_t, mask)
        noise = torch.randn_like(expectation, device=device)
        x_t = expectation + standard_deviation * noise
        x_t = self.masker.apply_mask(x_t, mask)
        return {"x_t": x_t, "noise": noise, "expectation": expectation, "standard_deviation": standard_deviation}

    def sample_xtm1_conditional_on_xt(
        self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
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
            Tensor: the sample at timestep t-1
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
        elif mode == "x_0":
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

        return pred_prev_sample

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

        def _posterior_mean_fn(
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
            if score is None:
                score = score_fn(x_t, t, padding_mask)
            config = cast(EuclideanDDPMConfig, self.config.to(t))
            alpha_bar_t = config.alphas_cumprod[t]  # macro_shape
            alpha_bar_t = self.complete_micro_shape(alpha_bar_t)
            x_0 = (x_t + (1 - alpha_bar_t) * score) / torch.sqrt(alpha_bar_t)
            return x_0

        return _posterior_mean_fn
