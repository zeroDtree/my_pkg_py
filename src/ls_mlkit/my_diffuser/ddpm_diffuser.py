"""Official package"""

import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple, Any
from tqdm.auto import tqdm

"""custom package"""
from .config import DiffusionConfig
from ls_mlkit.my_utils import MaskerInterface, BioCAOnlyMasker
from .loss_utils import rmsd
from .conditioner import Conditioner
from .model_interface import ModelInterface4Diffuser
from .diffuser import Diffuser


class DDPMDiffuser(Diffuser):
    def __init__(
        self,
        model: ModelInterface4Diffuser,
        config: DiffusionConfig,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = rmsd,
        masker: MaskerInterface = BioCAOnlyMasker(),
        conditioner_list: list[Conditioner] = [],
    ):
        super().__init__(
            model=model,
            config=config,
            loss_fn=loss_fn,
            masker=masker,
            conditioner_list=conditioner_list,
        )

    def sample_a_time_step(self, macro_shape: Tuple[int, ...]) -> Tensor:
        """
        Sample a timestep
        """
        continuous = self.config.continuous
        if continuous:
            raise NotImplementedError("Continuous time diffusion is not implemented")
        else:
            return torch.randint(0, self.config.n_discretization_steps, macro_shape)

    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        config = self.config
        batch = self.model.prepare_batch_data_for_input(batch)
        assert isinstance(batch, dict), "batch must be a dictionary"
        x_0 = batch["x_0"]
        padding_mask = batch["padding_mask"]
        device = x_0.device
        macro_shape = self.get_macro_shape(x_0)
        t = self.sample_a_time_step(macro_shape).to(device)
        loss = None

        conditioner_dict = {
            "x_0": x_0,
            "padding_mask": padding_mask,
        }

        diffusion_type = config.diffusion_type

        if diffusion_type == "DDPM":
            expectation, standard_deviation = self.q_xt_x_0(x_0, t, padding_mask)
            noise = torch.randn_like(expectation, device=device)
            x_t = expectation + standard_deviation * noise
            x_t = self.masker.apply_mask(x_t, padding_mask)
            predicted_noise = self.model(x_t, t, padding_mask)
            predicted_score = -predicted_noise
            ground_truth_unconditional_score = -noise

            conditioner_dict["unconditional_score"] = predicted_score / standard_deviation
            for conditioner in self.conditioner_list:
                tmp_conditioner_dict: dict[str, Any] = {
                    **conditioner_dict,
                    **conditioner.prepare_condition_dict(train=True, **conditioner_dict),
                }
                conditioner.set_condition(**tmp_conditioner_dict)

            accumulated_conditional_score = self.get_accumulated_conditional_score(x_t, t, padding_mask)
            ground_truth_score = (
                ground_truth_unconditional_score
                + (
                    self.get_something_proper_shape(
                        x_t,
                        config.sqrt_1m_alphas_cumprod.to(t.device)[t],
                    )
                )
                * accumulated_conditional_score
            )
            loss = self.loss_fn(predicted_score, ground_truth_score, padding_mask)

        elif diffusion_type == "VPSDE":
            pass
        elif diffusion_type == "VESDE":
            pass
        elif diffusion_type == "SubVPSDE":
            pass
        assert loss is not None, "loss must be computed"
        return loss

    def get_macro_and_micro_shape(self, x: Tensor):
        ndim_micro_shape = self.config.ndim_micro_shape
        return x.shape[:-ndim_micro_shape], x.shape[-ndim_micro_shape:]

    def get_macro_shape(self, x: Tensor):
        return x.shape[: -self.config.ndim_micro_shape]

    def get_something_proper_shape(self, base: Tensor, something: Tensor) -> Tensor:
        macro_shape = self.get_macro_shape(base)
        assert something.shape == macro_shape, "something.shape must be equal to x.macro_shape"
        something = something.view(*macro_shape, *[1 for _ in range(self.config.ndim_micro_shape)])
        return something.to(base.device)

    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward process
        $$q(x_t|x_0)$$
        return (expectation, standard_deviation)
        """
        expectation, standard_deviation = None, None
        diffusion_type = self.config.diffusion_type
        if diffusion_type == "DDPM":
            sqrt_alphas_cumprod = self.config.sqrt_alphas_cumprod.to(t.device)
            sqrt_1m_alphas_cumprod = self.config.sqrt_1m_alphas_cumprod.to(t.device)
            expectation = self.get_something_proper_shape(x_0, sqrt_alphas_cumprod[t]) * x_0
            standard_deviation = self.get_something_proper_shape(x_0, sqrt_1m_alphas_cumprod[t])
        elif diffusion_type == "VPSDE":
            pass
        elif diffusion_type == "VESDE":
            pass
        elif diffusion_type == "SubVPSDE":
            pass
        assert expectation is not None and standard_deviation is not None
        return expectation, standard_deviation

    def sample_xtm1_conditional_on_xt(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        r"""
        $$
        \hat{\mathbf{x}}_0:=\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon}_{\theta}(\mathbf{x}_t,t))
        $$

        $$
        \mathcal{N}\left( \boldsymbol{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\boldsymbol{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{\boldsymbol{x}}_0}{1-\bar{\alpha}_t}}_{\mu_q(\boldsymbol{x}_t, \hat{\boldsymbol{x}}_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\Sigma_q(t)} \right)
        $$


        """

        config = self.config
        for k, v in config.__dict__.items():
            if isinstance(v, Tensor):
                setattr(config, k, v.to(t.device))
        t = t.long()
        epsilon_predicted = self.model(x_t, t, padding_mask)

        prev_t = t - 1
        # 1. compute alphas, betas
        alpha_prod_t = config.alphas_cumprod[t]
        alpha_prod_t_prev = config.alphas_cumprod[prev_t] if prev_t >= 0 else torch.ones(1).to(t.device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (x_t - beta_prod_t ** (0.5) * epsilon_predicted) / alpha_prod_t ** (0.5)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t

        # 6. Add noise
        variance = torch.randn_like(x_t) * torch.sqrt(config.betas)[t]
        if t > 0:
            pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
