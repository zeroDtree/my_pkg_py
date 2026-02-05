from typing import Any, Callable, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from ..util.sde.corrector import LangevinCorrector
from ..util.sde.sde_lib import VPSDE
from .conditioner import Conditioner
from .conditioner.utils import get_accumulated_conditional_score
from .euclidean_diffuser import EuclideanDiffuser, EuclideanDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class EuclideanVPSDEConfig(EuclideanDiffuserConfig):
    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        use_probability_flow=False,
        beta_min: float = 0.1,
        beta_max: float = 20,
        n_correct_steps: int = 1,
        snr: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
        )

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.sde = VPSDE(
            beta_min=beta_min,
            beta_max=beta_max,
            ndim_micro_shape=ndim_micro_shape,
        )
        self.use_probability_flow = use_probability_flow
        self.n_correct_steps = n_correct_steps
        self.snr = snr


@inherit_docstrings
class EuclideanVPSDEDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        config: EuclideanVPSDEConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: MaskerInterface,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted, ground_true, padding_mask)
    ):
        """Initialize the EuclideanVPSDEDiffuser

        Args:
            config (EuclideanVPSDEConfig): the config of the diffuser
            time_scheduler (DiffusionTimeScheduler): the time scheduler of the diffuser
            masker (MaskerInterface): the masker of the diffuser
            model (Module): the model of the diffuser
            loss_fn (Callable[[Tensor, Tensor, Tensor], Tensor]): the loss function of the diffuser

        Returns:
            None
        """
        super().__init__(config=config, time_scheduler=time_scheduler, masker=masker)
        self.config: EuclideanVPSDEConfig = config
        self.sde = config.sde
        self.model = model
        self.loss_fn = loss_fn

        def score_fn(x: Tensor, t: Tensor, mask: Tensor) -> Tensor:
            return self.model(x, t.long(), mask)["x"]

        self.corrector = LangevinCorrector(
            sde=self.sde,
            score_fn=score_fn,
            snr=self.config.snr,
            n_steps=self.config.n_correct_steps,
            ndim_micro_shape=self.config.ndim_micro_shape,
        )

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        return self.sde.prior_sampling(shape)

    def forward_process(
        self, x_0: Tensor, discrete_t: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        t = self.time_scheduler.discrete_time_to_continuous_time(discrete_t)
        forward_result = self.sde.forward_process(x_0, t, mask)
        return {
            "x_t": forward_result["x_t"],
            "mean": forward_result["mean"],
            "std": forward_result["std"],
            "a": forward_result["a"],
            "b": forward_result["b"],
        }

    def compute_loss(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> dict:
        batch = self.model.prepare_batch_data_for_input(batch)
        assert isinstance(batch, dict), "batch must be a dictionary"
        x_0 = batch["gt_data"]
        padding_mask = batch["padding_mask"]
        device = x_0.device
        macro_shape = self.get_macro_shape(x_0)

        t = batch.get("t", None)
        if t is None:
            t = self.time_scheduler.sample_a_discrete_time_step_uniformly(macro_shape).to(device)
        self.config = self.config.to(t)

        forward_result = self.forward_process(x_0, t, padding_mask)
        x_t = forward_result["x_t"]
        mean = forward_result["mean"]
        std = forward_result["std"]
        a = forward_result["a"]
        b = forward_result["b"]
        gt_uc_score = self.sde.get_score(x_t=x_t, mean=mean, std=std)

        model_input_dict = batch
        model_input_dict.pop("gt_data")
        model_input_dict.pop("padding_mask")
        model_input_dict.pop("t", None)
        model_output = self.model(x_t, t, padding_mask, **model_input_dict)
        p_uc_score = model_output["x"]

        gt_uc_score = b * gt_uc_score
        p_uc_score = b * p_uc_score

        loss = self.loss_fn(p_uc_score, gt_uc_score, padding_mask)

        return {
            "loss": loss,
            "gt_data": x_0,
            "t": t,
            "x_t": x_t,
            "padding_mask": padding_mask,
            "gt_uc_score": gt_uc_score,
            "p_uc_score": p_uc_score,
            "a": a,
            "b": b,
            "loss_fn": self.loss_fn,
            "config": self.config,
        }

    def forward_process_n_step(
        self, x: Tensor, t: Tensor, next_t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        assert (next_t > t).all()
        assert (t >= 0).all()
        assert (next_t < self.config.n_discretization_steps).all()

        continuous_t1 = self.time_scheduler.discrete_time_to_continuous_time(t)
        continuous_t2 = self.time_scheduler.discrete_time_to_continuous_time(next_t)
        x_t2 = self.sde.forward_from_t1_to_t2(x, continuous_t1, continuous_t2)
        return x_t2

    def step(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
        r"""
        Args:
            x_t (Tensor): the sample at timestep t
            t (Tensor): the timestep
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the sample at timestep t-1
        """
        assert torch.all(t == t.view(-1)[0]).item()
        device = x_t.device
        idx = kwargs.get("idx")
        ones = torch.ones_like(t)
        t_start = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx] * ones
        t_end = self.time_scheduler.get_continuous_timesteps_schedule().to(device)[idx + 1] * ones
        config = cast(EuclideanVPSDEConfig, self.config.to(device))
        model_output = self.model(x_t, t.long(), padding_mask, *args, **kwargs)
        p_uc_score = model_output["x"]

        # score hook start=====================================================
        hook_input = {
            "p_uc_score": p_uc_score,
            "x_t": x_t,
            "t": t,
            "padding_mask": padding_mask,
            "config": config,
            "sampling_condition": kwargs.get("sampling_condition"),
        }
        hook_output = self.hook_manager.run_hooks(GMHookStageType.PRE_UPDATE_IN_STEP_FN, **hook_input)
        if hook_output is not None:
            p_uc_score = hook_output

        # score hook start end =================================================================

        rsde = self.sde.get_reverse_sde(
            score=p_uc_score, score_fn=None, use_probability_flow=self.config.use_probability_flow
        )
        delta_t = t_end - t_start
        delta_t = self.complete_micro_shape(delta_t)
        f, g = rsde.get_drift_and_diffusion(x_t, t_start, mask=padding_mask)
        g = self.complete_micro_shape(g)
        z = torch.randn_like(x_t)
        x_mean = x_t + f * delta_t
        if (t > 0).all():
            x = x_mean + g * z * torch.sqrt(delta_t.abs())
        else:
            x = x_mean

        if (t > 0).all():
            x, _ = self.corrector.update_fn(x, t - 1, padding_mask)

        return {
            "x": x,
        }

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

            For the case of VPSDE sampling, the posterior mean is given by

            .. math::

                E[x_0|x_t] = \frac{b^2}{a} \nabla_{x_t}\log p_t(x_t) - \frac{x_t}{a}

            """
            nonlocal score, score_fn
            assert score is not None or score_fn is not None, "either score or score_fn must be provided"
            if score is None:
                score = score_fn(x_t, t, padding_mask)
            sde = cast(EuclideanVPSDEConfig, self.config.to(t)).sde
            t = self.time_scheduler.discrete_time_to_continuous_time(t)
            a, b = sde.get_a_b(t)
            E_x0_xt = b**2 / a * score + x_t / a
            return E_x0_xt

        return _posterior_mean_fn

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):

        def _hook_fn(**kwargs):
            nonlocal conditioner_list

            loss = kwargs.get("loss")
            x_0 = kwargs.get("gt_data")
            x_t = kwargs.get("x_t")
            t = kwargs.get("t", None)
            padding_mask = kwargs.get("padding_mask")
            loss_fn = kwargs.get("loss_fn")
            config = kwargs.get("config")
            p_uc_score = kwargs.get("p_uc_score")
            gt_uc_score = kwargs.get("gt_uc_score")
            a = kwargs.get("a")
            b = kwargs.get("b")

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
            total_loss = loss_fn(p_uc_score, gt_score, padding_mask)
            kwargs["loss"] = total_loss
            return kwargs

        return GMHook(
            name="VPSDE_condition_post_compute_loss_hook",
            stage=GMHookStageType.POST_COMPUTE_LOSS,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )

    def get_condition_pre_update_in_step_fn_hook(self, conditioner_list: list[Conditioner]):
        def _hook_fn(**kwargs):
            nonlocal conditioner_list
            p_uc_score = kwargs.get("p_uc_score")
            x_t = kwargs.get("x_t")
            t = kwargs.get("t", None)
            padding_mask = kwargs.get("padding_mask")
            config = kwargs.get("config")
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
                                "posterior_mean_fn": self.get_posterior_mean_fn(score=p_uc_score, score_fn=None),
                            },
                        ),
                    }
                )

            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            p_score = p_uc_score + acc_c_score
            return p_score

        return GMHook(
            name="VPSDE_condition_pre_update_in_step_fn_hook",
            stage=GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            fn=_hook_fn,
            priority=0,
            enabled=True,
        )
