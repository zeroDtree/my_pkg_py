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


class Diffuser(Module):
    def __init__(
        self,
        model: ModelInterface4Diffuser,
        diffusion_config: DiffusionConfig,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = rmsd,
        masker: MaskerInterface = BioCAOnlyMasker(),
        conditioner_list: list[Conditioner] = [],
    ):
        super().__init__()  # type: ignore
        self.model: ModelInterface4Diffuser = model
        self.diffusion_config: DiffusionConfig = diffusion_config
        self.loss_fn = loss_fn
        self.masker = masker
        self.conditioner_list = conditioner_list

    def sample_a_time_step(self, macro_shape: Tuple[int, ...]) -> Tensor:
        """
        Sample a timestep
        """
        continuous = self.diffusion_config.continuous
        if continuous:
            raise NotImplementedError("Continuous time diffusion is not implemented")
        else:
            return torch.randint(0, self.diffusion_config.n_discretization_steps, macro_shape)

    def get_accumulated_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        accumulated_conditional_score = torch.zeros_like(x_t)
        for conditioner in self.conditioner_list:
            accumulated_conditional_score += conditioner.get_conditional_score(x_t, t, padding_mask)
        return accumulated_conditional_score

    def forward(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]):
        return self.compute_loss(batch, *args, **kwargs)

    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        diffusion_config = self.diffusion_config
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

        diffusion_type = diffusion_config.diffusion_type

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
                        diffusion_config.sqrt_1m_alphas_cumprod.to(t.device)[t],
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
        ndim_micro_shape = self.diffusion_config.ndim_micro_shape
        return x.shape[:-ndim_micro_shape], x.shape[-ndim_micro_shape:]

    def get_macro_shape(self, x: Tensor):
        return x.shape[: -self.diffusion_config.ndim_micro_shape]

    def get_something_proper_shape(self, base: Tensor, something: Tensor) -> Tensor:
        macro_shape = self.get_macro_shape(base)
        assert something.shape == macro_shape, "something.shape must be equal to x.macro_shape"
        something = something.view(*macro_shape, *[1 for _ in range(self.diffusion_config.ndim_micro_shape)])
        return something.to(base.device)

    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward process
        $$q(x_t|x_0)$$
        return (expectation, standard_deviation)
        """
        expectation, standard_deviation = None, None
        diffusion_type = self.diffusion_config.diffusion_type
        if diffusion_type == "DDPM":
            sqrt_alphas_cumprod = self.diffusion_config.sqrt_alphas_cumprod.to(t.device)
            sqrt_1m_alphas_cumprod = self.diffusion_config.sqrt_1m_alphas_cumprod.to(t.device)
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

    # def sample_xtm1_conditional_on_xt(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
    #     """Sample $$x_{t-1}$$ conditional on $$x_t$$
    #     $$p(x_{t-1}|x_t)$$
    #     """
    #     diffusion_config = self.diffusion_config
    #     diffusion_type = diffusion_config.diffusion_type
    #     model: ModelInterface4Diffuser = self.model
    #     masker = self.masker
    #     x_tm1 = None
    #     if diffusion_type == "DDPM":
    #         r"""
    #         $$
    #         \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sqrt{\beta_t}\mathbf{\epsilon}
    #         $$
    #         """
    #         t = t.long()
    #         coefficient = (1.0 - diffusion_config.alphas.to(t.device)[t]) / torch.sqrt(
    #             1 - diffusion_config.alphas_cumprod.to(t.device)[t]
    #         )
    #         coefficient = self.get_something_proper_shape(x_t, coefficient)
    #         noise_predicted = model(x_t, t, padding_mask)
    #         expectation = self.get_something_proper_shape(
    #             x_t, 1.0 / torch.sqrt(diffusion_config.alphas.to(t.device))[t]
    #         ) * (x_t - coefficient * noise_predicted)
    #         expectation = masker.apply_mask(expectation, padding_mask)
    #         if (t == 0).all() and diffusion_config.denoise_at_final:
    #             return expectation
    #         else:
    #             prior_noise = torch.randn_like(expectation).to(x_t.device)
    #             standard_deviation = self.get_something_proper_shape(
    #                 x_t,
    #                 torch.sqrt(diffusion_config.betas.to(t.device)[t]),
    #             )
    #             x_tm1 = expectation + standard_deviation * prior_noise
    #             x_tm1 = masker.apply_mask(x_tm1, padding_mask)

    #     elif diffusion_type == "VPSDE":
    #         pass
    #     elif diffusion_type == "VESDE":
    #         pass
    #     elif diffusion_type == "SubVPSDE":
    #         pass

    #     assert x_tm1 is not None
    #     return x_tm1

    @torch.no_grad()  # type: ignore
    def sample_x0_unconditionally(self, shape: Tuple[int, ...]):
        r"""
        $$
        P_0 \leftarrow  P_1 \leftarrow \cdots \leftarrow P_T
        $$
        """
        macro_shape = shape[: -self.diffusion_config.ndim_micro_shape]
        diffusion_config = self.diffusion_config
        masker = self.masker
        model = self.model
        device = model.get_model_device()
        x_t = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(diffusion_config.n_discretization_steps))):
            t = i
            t = torch.ones(macro_shape, device=device) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            x_t = self.sample_xtm1_conditional_on_xt(x_t, t, no_padding_mask)

        return x_t

    @torch.no_grad()  # type: ignore
    def inpainting_x0_unconditionally(self, x_0: Tensor, padding_mask: Tensor, inpainting_mask: Tensor):
        r"""
        $$
        P_0 \leftarrow  P_1 \leftarrow \cdots \leftarrow P_T
        $$
        """
        shape = x_0.shape
        diffusion_config = self.diffusion_config
        macro_shape = shape[: -diffusion_config.ndim_micro_shape]
        masker = self.masker
        model: ModelInterface4Diffuser = self.model
        device = model.get_model_device()

        x_t = torch.randn(shape, device=device)
        x_0 = masker.apply_mask(x_0, padding_mask)
        x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)
        for i in tqdm(reversed(range(diffusion_config.n_discretization_steps))):
            t = i
            t = torch.ones(macro_shape, device=device) * t
            x_t = self.sample_xtm1_conditional_on_xt(x_t, t, padding_mask)
            x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)
            x_t = masker.apply_mask(x_t, padding_mask)

        return x_t

    def sample_xtm1_conditional_on_xt(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        r"""
        $$
        \hat{\mathbf{x}}_0:=\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon}_{\theta}(\mathbf{x}_t,t))
        $$

        $$
        \mathcal{N}\left( \boldsymbol{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})\boldsymbol{x}_t + \sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{\boldsymbol{x}}_0}{1-\bar{\alpha}_t}}_{\mu_q(\boldsymbol{x}_t, \hat{\boldsymbol{x}}_0)}, \underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\Sigma_q(t)} \right)
        $$


        """

        config = self.diffusion_config
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
