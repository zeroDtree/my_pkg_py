r"""
Riemannian Manifold Diffuser
"""

from typing import Any

from ..util.decorators import inherit_docstrings
from ..util.manifold.riemannian_manifold import RiemannianManifold
from .base_diffuser import BaseDiffuser, BaseDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class RiemannianManifoldDiffuserConfig(BaseDiffuserConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: int,
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


@inherit_docstrings
class RiemannianManifoldDiffuser(BaseDiffuser):
    def __init__(
        self,
        config: RiemannianManifoldDiffuserConfig,
        time_scheduler: DiffusionTimeScheduler,
        riemannian_manifold: RiemannianManifold,
    ):
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
        )
        self.riemannian_manifold = riemannian_manifold
