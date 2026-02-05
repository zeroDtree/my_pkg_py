from typing import Any, Callable, Literal, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from .euclidean_ddpm_diffuser import EuclideanDDPMConfig, EuclideanDDPMDiffuser
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class EuclideanDDIMConfig(EuclideanDDPMConfig):
    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        use_probability_flow=False,
        use_clip: bool = True,
        clip_sample_range: float = 1.0,
        use_dyn_thresholding: bool = False,
        dynamic_thresholding_ratio=0.995,
        sample_max_value: float = 1.0,
        betas=None,
        n_inference_steps: int = 1000,
        eta: float = 0.0,
        *args,
        **kwargs,
    ):
        """Initialize the EuclideanDDIMConfig

        Args:
            n_discretization_steps (int): the number of discretization steps
            ndim_micro_shape (int): the number of dimensions of the micro shape
            use_probability_flow (bool): whether to use probability flow
            use_clip (bool): whether to use clip
            clip_sample_range (float): the range of the clip
            use_dyn_thresholding (bool): whether to use dynamic thresholding
            dynamic_thresholding_ratio (float): the ratio of the dynamic thresholding
            sample_max_value (float): the maximum value of the sample used in thresholding
            betas (Tensor): the betas
            n_inference_steps (int): the number of inference steps
            eta (float): the eta

        Returns:
            None
        """
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
            use_probability_flow=use_probability_flow,
            use_clip=use_clip,
            clip_sample_range=clip_sample_range,
            use_dyn_thresholding=use_dyn_thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
            betas=betas,
        )
        self.n_inference_steps = n_inference_steps
        self.eta: float = eta


@inherit_docstrings
class EuclideanDDIMDiffuser(EuclideanDDPMDiffuser):
    def __init__(
        self,
        config: EuclideanDDPMConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: MaskerInterface,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted, ground_true, padding_mask)
    ):
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
            masker=masker,
            model=model,
            loss_fn=loss_fn,
        )

    def get_sigma2(self, t: Tensor, prev_t: Tensor) -> Tensor:
        r"""Compute DDIM variance term

        .. math::
            \sigma^2 = (\frac{1 - \bar{\alpha}_{pre}}{1 - \bar{\alpha}_{t}}) \cdot ( 1- \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{pre}})

        Args:
            t (Tensor): timestep
            prev_t (Tensor): previous timestep

        Returns:
            Tensor: :math:`\sigma^2`
        """
        config = cast(EuclideanDDIMConfig, self.config)
        alpha_prod_t = config.alphas_cumprod[t]
        alpha_prod_t_prev = config.alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones(1).to(t.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # DDIM variance formula
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
        r"""DDIM sampling algorithm:

        .. math::

            \hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}

            \text{direction} = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)

            x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \text{direction} + \sigma_t \cdot z

        Args:
            x_t (Tensor): the sample at timestep t
            t (Tensor): the timestep
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the sample at timestep t-1
        """
        assert torch.all(t == t.view(-1)[0]).item()
        config = cast(EuclideanDDIMConfig, self.config.to(t))
        t = t.long()
        t = t.view(-1)[0]
        # DDIM requires proper timestep scaling for inference
        # When using fewer inference steps than training steps, we need to scale the timestep difference
        step_ratio = config.n_discretization_steps // config.n_inference_steps
        prev_t = t - step_ratio
        alpha_prod_t = config.alphas_cumprod[t]
        alpha_prod_t_prev = config.alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones(1).to(t.device)
        beta_prod_t = 1 - alpha_prod_t

        mode: Literal["epsilon", "x_0", "score"] = kwargs.get("mode", "epsilon")
        # print(f"mode: {mode}, t={t}, prev_t={prev_t}")
        if mode == "epsilon":
            epsilon_predicted = self.model(x_t, t, padding_mask, *args, **kwargs)["x"]
        elif mode == "x_0":
            p_x_0 = self.model(x_t, t, padding_mask, *args, **kwargs)["x"]
            epsilon_predicted = (x_t - alpha_prod_t ** (0.5) * p_x_0) / beta_prod_t ** (0.5)
        elif mode == "score":
            raise ValueError(f"Currently not supported mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        r"""
        $$\hat{x_0} = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$
        """
        pred_original_sample = None
        if mode in ["epsilon"]:
            pred_original_sample = (x_t - beta_prod_t ** (0.5) * epsilon_predicted) / alpha_prod_t ** (0.5)
        elif mode in ["x_0"]:
            pred_original_sample = p_x_0

        r"""
        $$\sigma = \eta \cdot \sqrt{(\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}}) \cdot ( 1- \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{t-1}})}$$
        """
        sigma2 = self.get_sigma2(t, prev_t)
        sigma = config.eta * torch.sqrt(sigma2)

        r"""
        $$direction = \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t)$$
        """
        direction = torch.sqrt(1 - alpha_prod_t_prev - sigma**2) * epsilon_predicted

        r"""
        $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \text{direction} + \sigma_t \cdot z$$
        """
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + direction

        epsilon_t = torch.randn_like(x_t)
        if t > 0:
            pred_prev_sample = pred_prev_sample + sigma * epsilon_t

        return {"x": pred_prev_sample, "E_x0_xt": pred_original_sample}
