"""Official package"""

import torch
from torch import Tensor
from typing import Callable, Tuple, Any, cast

"""custom package"""
from .ddpm_config import DDPMConfig
from ls_mlkit.my_utils import MaskerInterface, BioCAOnlyMasker
from .loss_utils import rmsd
from .conditioner import Conditioner
from .model_interface import ModelInterface4Diffuser
from .diffuser import Diffuser


class DDPMDiffuserV2(Diffuser):
    def __init__(
        self,
        model: ModelInterface4Diffuser,
        config: DDPMConfig,
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

    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        config = self.config
        batch = self.model.prepare_batch_data_for_input(batch)
        assert isinstance(batch, dict), "batch must be a dictionary"
        x_0 = batch["x_0"]
        padding_mask = batch["padding_mask"]
        device = x_0.device
        macro_shape = self.get_macro_shape(x_0)
        t = self.sample_a_time_step(macro_shape).to(device)
        config = config.to(t)
        loss = None

        conditioner_dict = {
            "x_0": x_0,
            "padding_mask": padding_mask,
        }

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
                    cast(DDPMConfig, self.config).sqrt_1m_alphas_cumprod[t],
                )
            )
            * accumulated_conditional_score
        )
        loss = self.loss_fn(predicted_score, ground_truth_score, padding_mask)
        return loss

    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward process
        $$q(x_t|x_0)$$
        return (expectation, standard_deviation)
        """
        config = cast(DDPMConfig, self.config.to(t))
        expectation = self.get_something_proper_shape(x_0, config.sqrt_alphas_cumprod[t]) * x_0
        standard_deviation = self.get_something_proper_shape(x_0, config.sqrt_1m_alphas_cumprod[t])
        return expectation, standard_deviation

    def sample_xtm1_conditional_on_xt(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        """Sample $$x_{t-1}$$ conditional on $$x_t$$
        $$p(x_{t-1}|x_t)$$
        """
        config = cast(DDPMConfig, self.config.to(t))
        model = self.model
        masker = self.masker
        x_tm1 = None
        r"""
        $$
        \mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sqrt{1-\alpha_t}\mathbf{\epsilon}
        $$
        """
        t = t.long()
        coefficient = (1.0 - config.alphas[t]) / torch.sqrt(1 - config.alphas_cumprod[t])
        coefficient = self.get_something_proper_shape(x_t, coefficient)
        noise_predicted = model(x_t, t, padding_mask)
        expectation = self.get_something_proper_shape(
            x_t,
            (1.0 / torch.sqrt(config.alphas[t])),
        ) * (x_t - coefficient * noise_predicted)
        expectation = masker.apply_mask(expectation, padding_mask)
        if (t == 0).all() and config.denoise_at_final:
            return expectation
        else:
            prior_noise = torch.randn_like(expectation).to(t.device)
            standard_deviation = self.get_something_proper_shape(x_t, config.betas[t].sqrt())
            x_tm1 = expectation + standard_deviation * prior_noise
            x_tm1 = masker.apply_mask(x_tm1, padding_mask)

        return x_tm1
