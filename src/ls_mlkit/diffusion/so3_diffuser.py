from typing import Any, Callable, Tuple

import torch
from torch import Tensor

from ..util.decorators import inherit_docstrings
from ..util.interp import interp
from ..util.manifold.so3 import SO3
from ..util.manifold.so3_utils import (
    calculate_igso3,
    inverse_transform_sampling,
    rotation_matrix_to_angle,
    vector_to_skew_symmetric,
)
from ..util.mask.masker import Masker as BioSO3Masker
from ..util.sde.base_sde import SDE
from ..util.sde.sde_lib import VESDE
from .lie_group_diffuser import LieGroupDiffuser, LieGroupDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler

EPS = 1e-6


@inherit_docstrings
class SO3DiffuserConfig(LieGroupDiffuserConfig):

    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: int,
        igso3_num_sigma: int,
        igso3_num_omega: int,
        igso3_min_sigma: float,
        igso3_max_sigma: float,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        super().__init__(
            ndim_micro_shape=ndim_micro_shape,
            n_discretization_steps=n_discretization_steps,
            n_inference_steps=n_inference_steps,
            *args,
            **kwargs,
        )

        self.igso3_num_sigma = igso3_num_sigma
        self.igso3_num_omega = igso3_num_omega
        self.igso3_min_sigma = igso3_min_sigma
        self.igso3_max_sigma = igso3_max_sigma


@inherit_docstrings
class SO3Diffuser(LieGroupDiffuser):

    def __init__(
        self,
        config: SO3DiffuserConfig,
        time_scheduler: DiffusionTimeScheduler,
        masker: BioSO3Masker,
        sde: SDE,
        score_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (x, t, mask) -> score
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # (predicted_score, ground_truth_score, mask) -> loss
    ):
        so3 = SO3()
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
            lie_group=so3,
        )
        self.config = config
        self.time_scheduler = time_scheduler
        self.masker = masker
        self.sde = sde
        self.loss_fn = loss_fn
        self.so3 = so3
        self.score_fn = score_fn
        assert isinstance(self.sde, VESDE), "only VESDE is supported"
        igso3_cache = calculate_igso3(
            num_sigma=config.igso3_num_sigma,
            num_omega=config.igso3_num_omega,
            min_sigma=config.igso3_min_sigma,
            max_sigma=config.igso3_max_sigma,
            discrete_omega=torch.linspace(0, torch.pi, config.igso3_num_omega + 1)[1:],
            discrete_sigma=self.sde.discrete_sigmas,
        )

        # Register buffers - these will automatically move with the model
        self.register_buffer("_igso3_cdf", igso3_cache["cdf"])  # [num_sigma, num_omega]
        self.register_buffer(
            "_igso3_score_norm", igso3_cache["score_norm"]
        )  # [num_sigma, num_omega] # $$\frac{d}{d\omega} f(\omega, c, L)$$

        self.register_buffer(
            "_igso3_exp_score_norms", igso3_cache["exp_score_norms"]
        )  # [num_sigma, ]                  $$\sqrt{\mathbb{E}_{\omega} || \frac{d}{d\omega} f(\omega, c, L)||_2^2}$$

        self.register_buffer("_igso3_discrete_omega", igso3_cache["discrete_omega"])  # [num_omega, ]
        self.register_buffer("_igso3_discrete_sigma", igso3_cache["discrete_sigma"])  # [num_sigma, ]

    @property
    def igso3_cdf(self) -> Tensor:
        return self._igso3_cdf

    @property
    def igso3_score_norm(self) -> Tensor:
        return self._igso3_score_norm

    @property
    def igso3_exp_score_norms(self) -> Tensor:
        return self._igso3_exp_score_norms

    @property
    def igso3_discrete_omega(self) -> Tensor:
        return self._igso3_discrete_omega

    @property
    def igso3_discrete_sigma(self) -> Tensor:
        return self._igso3_discrete_sigma

    def prior_sampling(self, shape: Tuple[int, ...]) -> Tensor:
        r"""Sample initial noise used for reverse process

        .. math::

            \mathcal{U}_{SO(3)}

        Args:
            shape (Tuple[int, ...]): the shape of the sample

        Returns:
            Tensor: the initial noise
        """
        macro_shape = shape
        discrete_t = self.time_scheduler.num_train_timesteps - 1
        axis = torch.randn(macro_shape + (3,))
        axis_in_s2 = axis / torch.norm(axis, dim=-1, keepdim=True)
        angle = inverse_transform_sampling(
            shape=macro_shape, cdf=self.igso3_cdf[discrete_t], discrete_omega=self.igso3_discrete_omega
        )
        rotation_vector = angle * axis_in_s2
        rotation_skew_symmetric = vector_to_skew_symmetric(rotation_vector)
        rotation_matrix = self.so3.exp(v=rotation_skew_symmetric)
        return rotation_matrix

    def forward_process(
        self, x_0: Tensor, discrete_t: Tensor, mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        r"""Forward process

        .. math::

            \text{IG}_{\text{SO}(3)} (\mathbf{x}; \mathbf{\mu}, \sigma^2) = f_{\sigma} (\arccos((\text{tr}(\mathbf{\mu}^T \mathbf{x}) - 1)/2)) \quad \forall \mathbf{x} \in \text{SO}(3)

        Args:
            x_0 (Tensor): the initial sample
            discrete_t (Tensor): the discrete timestep
            mask (Tensor): the mask
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            dict: a dictionary that must contain the key "x_t"
        """
        # x.shape = (b, n, 3, 3)
        macro_shape = self.get_macro_shape(x_0)  # (*macro_shape, ) = (b,)
        n = x_0.shape[-3]
        shape = macro_shape + (n,)
        device = x_0.device
        axis = torch.randn(shape + (3,), device=device)  # (*macro_shape, n, 3)
        axis_in_s2 = axis / torch.norm(axis, dim=-1, keepdim=True)  # (*macro_shape, n, 3)
        igso3_cdf = self.igso3_cdf[discrete_t]  # (*macro_shape, num_omega)
        igso3_cdf = igso3_cdf.unsqueeze(-2).expand(*macro_shape, n, -1)  # (*macro_shape, n, num_omega)
        angle = inverse_transform_sampling(
            shape=shape, cdf=igso3_cdf, discrete_omega=self.igso3_discrete_omega
        )  # (*macro_shape, n)
        rotation_vector = angle.unsqueeze(-1) * axis_in_s2  # (*macro_shape, n, 3)
        rotation_skew_symmetric = vector_to_skew_symmetric(rotation_vector)  # (*macro_shape,n 3, 3)
        rotation_matrix = self.so3.exp(v=rotation_skew_symmetric)  # (*macro_shape,n, 3, 3)
        x_t = self.so3.multiply(rotation_matrix, x_0)  # (*macro_shape, n, 3, 3)
        return {"x_t": x_t}

    def get_ground_truth_score(self, x_0: Tensor, x_t: Tensor, discrete_t: Tensor, padding_mask: Tensor) -> Tensor:
        """Denoise Score Matching

        .. math::
            \nabla_x \log p_{0t} (x_t | x_0)

        Args:
            x_0 (Tensor): _description_
            x_t (Tensor): _description_
            discrete_t (Tensor): _description_
            padding_mask (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        macro_shape = self.get_macro_shape(x_0)
        n = x_0.shape[-3]
        x_0t = x_0.transpose(-1, -2) @ x_t  # (*macro_shape, n, 3, 3)
        omega = rotation_matrix_to_angle(x_0t)  # (*macro_shape, n)
        igso3_score_norm = self.igso3_score_norm[discrete_t]  # (*macro_shape, num_omega)
        igso3_score_norm = igso3_score_norm.unsqueeze(-2).expand(*macro_shape, n, -1)  # (*macro_shape, n, num_omega)
        ground_truth_score = (
            x_t  # (*macro_shape, n, 3, 3)
            @ (self.so3.log(q=x_0t) / (omega.unsqueeze(-1).unsqueeze(-1) + EPS))  # (*macro_shape, n, 3, 3)
            * interp(x=omega, xp=self.igso3_discrete_omega, fp=igso3_score_norm)
            .unsqueeze(-1)
            .unsqueeze(-1)  # (*macro_shape, n, 3, 3)
        )
        return ground_truth_score

    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor:
        x_0 = batch["x_0"]
        padding_mask = batch["padding_mask"]
        macro_shape = self.get_macro_shape(x_0)
        discrete_t = batch.get("t", None)
        if discrete_t is None:
            discrete_t = self.time_scheduler.sample_a_discrete_time_step_uniformly(macro_shape=macro_shape)
        x_t = self.forward_process(x_0, discrete_t=discrete_t, mask=padding_mask)["x_t"]
        ground_truth_score = self.get_ground_truth_score(x_0, x_t, discrete_t, padding_mask)
        predicted_score = self.score_fn(x_t, discrete_t, padding_mask)
        loss = self.loss_fn(predicted_score, ground_truth_score, padding_mask)
        return loss

    def step(
        self, x_t: Tensor, discrete_t: Tensor, padding_mask: Tensor, *args: list[Any], **kwargs: dict[Any, Any]
    ) -> dict:
        r"""
        .. math::

            dx &= \exp_{x_t}(f_{rev} dt + g_{rev} dw)\\
            x_{t+\Delta_t} &= \exp_{x_t}(- f_{rev} |\Delta_t| + g_{rev} \Delta w)\\
            f_{rev} &= (f - g^2 \nabla_x \ln p_t(x))\\
            g_{rev} &= g\\

        """
        continuous_t = self.time_scheduler.discrete_time_to_continuous_time(discrete_t)
        f, g = self.sde.get_drift_and_diffusion(x=x_t, t=continuous_t, mask=padding_mask)

        # p_x_0: Tensor = kwargs.get("p_x_0", None)
        # assert p_x_0 is not None, "p_x_0 is required"
        # riemannian_grad = self.get_ground_truth_score(
        #     x_0=p_x_0, x_t=x_t, discrete_t=discrete_t, padding_mask=padding_mask
        # )

        riemannian_grad = self.score_fn(x_t, discrete_t, padding_mask)

        assert f.sum() == 0, "f should be 0"
        rev_f = f - g**2 * riemannian_grad
        rev_g = g

        delta_t = self.time_scheduler.T / self.time_scheduler.num_inference_timesteps
        term1 = -rev_f * delta_t

        noise_lie_algebra = self.sample_noise_in_lie_algebra(macro_shape=self.get_macro_shape(x_t))
        delta_w = torch.sqrt(delta_t) * x_t @ noise_lie_algebra
        term2 = rev_g * delta_w

        move_in_tangent_space = term1 + term2
        x_tm1 = self.so3.exp(p=x_t, v=move_in_tangent_space)
        return {"x": x_tm1}

    def sample_noise_in_lie_algebra(
        self,
        macro_shape: Tuple[int, ...],
    ) -> Tensor:
        r"""Sample noise in Lie algebra, Skew-symmetric matrix

        Args:
            macro_shape (Tuple[int, ...]): the macro shape of the noise

        Returns:
            Tensor: the noise in Lie algebra of shape :math:`(*macro_shape, 3, 3)`
        """
        return self.so3.random_tangent(p=self.so3.identity(macro_shape=macro_shape))

    def sampling(self, shape, device, x_init_posterior=None, *args, **kwargs):
        raise NotImplementedError

    def inpainting(
        self,
        x,
        padding_mask,
        inpainting_mask,
        device,
        x_init_posterior=None,
        inpainting_mask_key="inpainting_mask",
        *args,
        **kwargs,
    ):
        raise NotImplementedError
