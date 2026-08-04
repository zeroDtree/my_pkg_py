from typing import Any, Callable, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_gm_class import GMHook, GMHookStageType
from ..util.base_class.loss_mask import resolve_loss_mask
from ..util.context.temp_remove import TemporaryKeyRemover
from ..util.decorators import inherit_docstrings
from ..util.mask.masker_interface import MaskerInterface
from ..util.sde.corrector import LangevinCorrector
from ..util.sde.sde_lib import VPSDE
from ..util.typing_utils import require
from .conditioner import Conditioner
from .conditioner.utils import get_accumulated_conditional_score
from .euclidean_diffuser import EuclideanDiffuser, EuclideanDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class EuclideanVPSDEConfig(EuclideanDiffuserConfig):
    """
    Config Class for Euclidean VPSDE Diffuser
    """

    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        use_probability_flow=False,
        beta_min: float = 0.1,
        beta_max: float = 20,
        n_correct_steps: int = 1,
        snr: float = 1.0,
        model_uses_continuous_time: bool = False,
        *args,
        **kwargs,
    ):
        r"""
        Args:
            n_discretization_steps: the number of discretization steps
            ndim_micro_shape: the number of dimensions of the micro shape
            use_probability_flow: whether to use the probability flow ODE instead of the SDE
            beta_min: minimum beta value for the linear noise schedule
            beta_max: maximum beta value for the linear noise schedule
            n_correct_steps: number of Langevin corrector steps per denoising step
            snr: signal-to-noise ratio for the Langevin corrector
            model_uses_continuous_time: if True, pass continuous time in [0, 1] to the
                backbone (e.g. MLP / flow-style models); if False, pass discrete timestep
                indices (e.g. diffusers UNet)

        Returns:
            None
        """
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
        self.model_uses_continuous_time = model_uses_continuous_time


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
            return self.model(**{"x_t": x, "t": self._model_timestep_input(t), "padding_mask": mask})["x"]

        self.corrector = LangevinCorrector(
            sde=self.sde,
            score_fn=score_fn,
            snr=self.config.snr,
            n_steps=self.config.n_correct_steps,
            ndim_micro_shape=self.config.ndim_micro_shape,
        )

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        return self.sde.prior_sampling(shape)

    def _model_timestep_input(self, discrete_t: Tensor) -> Tensor:
        """Convert internal discrete timesteps to the format expected by the backbone."""
        if self.config.model_uses_continuous_time:
            flat_t = discrete_t.reshape(-1)
            continuous_t = self.time_scheduler.timestep_index_to_continuous_time(flat_t)
            return continuous_t.reshape(discrete_t.shape)
        return discrete_t.long()

    def forward_process(
        self,
        x_0: Tensor,
        discrete_t: Tensor,
        mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        t = self.time_scheduler.timestep_index_to_continuous_time(discrete_t)
        forward_result = self.sde.forward_process(x_0, t, mask)
        return {
            "x_t": forward_result["x_t"],
            "mean": forward_result["mean"],
            "std": forward_result["std"],
            "a": forward_result["a"],
            "b": forward_result["b"],
        }

    def compute_loss(self, **batch: Any) -> dict:
        """Compute the VPSDE score-matching loss.

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

        macro_shape = self.get_macro_shape(x_0)
        macro_shape = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_GET_MACRO_SHAPE,
            tgt_key_name="macro_shape",
            macro_shape=macro_shape,
            batch=batch,
        )
        macro_shape = cast(tuple[int, ...], macro_shape)

        t = batch.get("t", None)
        if t is None:
            t = self.time_scheduler.sample_timestep_index_uniformly(macro_shape).to(device)
        t = self.hook_manager.run_hooks(
            stage=GMHookStageType.POST_SAMPLING_TIME_STEP,
            tgt_key_name="t",
            t=t,
            batch=batch,
        )
        t = cast(Tensor, t)
        self.config = cast(EuclideanVPSDEConfig, self.config.to(t))

        forward_result = self.forward_process(x_0, t, padding_mask)
        x_t = forward_result["x_t"]
        mean = forward_result["mean"]
        std = forward_result["std"]
        a = forward_result["a"]
        b = forward_result["b"]
        gt_uc_score = self.sde.get_score(x_t=x_t, mean=mean, std=std)

        batch["x_t"] = x_t
        model_batch = {**batch, "t": self._model_timestep_input(t)}
        with TemporaryKeyRemover(mapping=model_batch, keys=["gt_data"]):
            model_output = self.model(**model_batch)
        p_uc_score = model_output["x"]

        gt_uc_score = b * gt_uc_score
        p_uc_score = b * p_uc_score

        loss_mask = resolve_loss_mask(self.hook_manager, padding_mask=padding_mask, batch=batch)
        loss = self.loss_fn(p_uc_score, gt_uc_score, loss_mask)

        return {
            "loss": loss,
            "gt_data": x_0,
            "t": t,
            "x_t": x_t,
            "padding_mask": padding_mask,
            "loss_mask": loss_mask,
            "gt_uc_score": gt_uc_score,
            "p_uc_score": p_uc_score,
            "a": a,
            "b": b,
            "loss_fn": self.loss_fn,
            "config": self.config,
            "base_model_output": model_output,
            "batch": batch,
        }

    def forward_process_n_step(
        self,
        x: Tensor,
        t: Tensor,
        next_t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        assert (next_t > t).all()
        assert (t >= 0).all()
        assert (next_t < self.config.n_discretization_steps).all()

        continuous_t1 = self.time_scheduler.timestep_index_to_continuous_time(t)
        continuous_t2 = self.time_scheduler.timestep_index_to_continuous_time(next_t)
        x_t2 = self.sde.forward_from_t1_to_t2(x, continuous_t1, continuous_t2)
        return x_t2

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        r"""
        Args:
            x_t (Tensor): the sample at timestep t
            t (Tensor): the timestep
            padding_mask (Tensor): the padding mask

        Returns:
            dict:
                - x: the sample at timestep t-1
        """
        assert torch.all(t == t.view(-1)[0]).item()
        device = x_t.device
        idx = require(kwargs.get("idx"), "idx")
        schedule = self.time_scheduler.get_continuous_boundaries_schedule().to(device)
        ones = torch.ones(x_t.shape[0], device=device)
        t_start = schedule[int(idx)] * ones
        t_end = schedule[int(idx) + 1] * ones
        config = cast(EuclideanVPSDEConfig, self.config.to(device))
        model_output = self.model(
            **{"x_t": x_t, "t": self._model_timestep_input(t), "padding_mask": padding_mask, **kwargs}
        )
        p_uc_score = model_output["x"]

        hook_input = {
            "p_uc_score": p_uc_score,
            "x_t": x_t,
            "t": t,
            "padding_mask": padding_mask,
            "config": config,
            "sampling_condition": kwargs.get("sampling_condition"),
        }
        hook_output = self.hook_manager.run_hooks(
            GMHookStageType.PRE_UPDATE_IN_STEP_FN,
            tgt_key_name="p_uc_score",
            **hook_input,
        )
        if hook_output is not None:
            p_uc_score = hook_output

        rsde = self.sde.get_reverse_sde(
            score=p_uc_score,
            score_fn=None,
            use_probability_flow=self.config.use_probability_flow,
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

    def get_posterior_mean_fn(
        self,
        score: Tensor | None = None,
        score_fn: Callable[[Tensor, Tensor, Tensor | None], Tensor] | None = None,
        batch: Optional[dict] = None,
    ):
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

            $$
            E[x_0|x_t] = \frac{b^2}{a} \nabla_{x_t}\log p_t(x_t) - \frac{x_t}{a}
            $$

            """
            nonlocal score, score_fn
            assert score is not None or score_fn is not None, "either score or score_fn must be provided"
            if score is None:
                assert score_fn is not None
                score = score_fn(x_t, t, padding_mask)
            sde = cast(EuclideanVPSDEConfig, self.config.to(t)).sde
            # Flatten t to 1D (batch only) so get_a_b adds exactly ndim_micro_shape dims.
            t_cont = self.time_scheduler.timestep_index_to_continuous_time(t.reshape(t.shape[0]))
            a, b = sde.get_a_b(t_cont)
            E_x0_xt = b**2 / a * score + x_t / a
            return E_x0_xt

        return _posterior_mean_fn

    def get_condition_post_compute_loss_hook(self, conditioner_list: list[Conditioner]):
        """Get hook for conditioning after loss computation (training).

        This hook modifies the loss to include conditional guidance during training.
        It computes the conditional score and updates the loss accordingly.

        Args:
            conditioner_list: list of conditioners

        Returns:
            GMHook: the hook for POST_COMPUTE_LOSS stage
        """

        def _hook_fn(**kwargs: Any):
            x_0 = require(cast(Tensor | None, kwargs.get("gt_data")), "gt_data")
            x_t = require(cast(Tensor | None, kwargs.get("x_t")), "x_t")
            t = require(cast(Tensor | None, kwargs.get("t")), "t")
            padding_mask = require(cast(Tensor | None, kwargs.get("padding_mask")), "padding_mask")
            loss_fn = require(cast(Callable[..., Any] | None, kwargs.get("loss_fn")), "loss_fn")
            p_uc_score = require(cast(Tensor | None, kwargs.get("p_uc_score")), "p_uc_score")
            gt_uc_score = require(cast(Tensor | None, kwargs.get("gt_uc_score")), "gt_uc_score")
            b = require(cast(Tensor | None, kwargs.get("b")), "b")

            self._setup_conditioners(
                conditioner_list,
                train=True,
                tgt_mask=padding_mask,
                padding_mask=padding_mask,
                p_uc_score=p_uc_score,
                gt_data=x_0,
            )
            acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
            gt_score = gt_uc_score + acc_c_score

            p_uc_score = b * p_uc_score
            gt_score = b * gt_score
            loss_mask = kwargs.get("loss_mask", padding_mask)
            total_loss = loss_fn(p_uc_score, gt_score, loss_mask)
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
        """Get hook for conditioning before update in step function (sampling).

        This hook applies conditional guidance during sampling by modifying
        the predicted score based on the conditional score.

        Args:
            conditioner_list: list of conditioners

        Returns:
            GMHook: the hook for PRE_UPDATE_IN_STEP_FN stage
        """

        def _hook_fn(**kwargs: Any):
            p_uc_score = require(cast(Tensor | None, kwargs.get("p_uc_score")), "p_uc_score")
            x_t = require(cast(Tensor | None, kwargs.get("x_t")), "x_t")
            t = require(cast(Tensor | None, kwargs.get("t")), "t")
            padding_mask = require(cast(Tensor | None, kwargs.get("padding_mask")), "padding_mask")
            sampling_condition = kwargs.get("sampling_condition")

            self._setup_conditioners(
                conditioner_list,
                train=False,
                tgt_mask=padding_mask,
                padding_mask=padding_mask,
                p_uc_score=p_uc_score,
                sampling_condition=sampling_condition,
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
