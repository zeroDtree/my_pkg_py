r"""
Lie Group Diffuser
"""

from typing import Any

from ..util.decorators import inherit_docstrings
from ..util.manifold.lie_group import LieGroup
from .manifold_diffuser import RiemannianManifoldDiffuser, RiemannianManifoldDiffuserConfig
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class LieGroupDiffuserConfig(RiemannianManifoldDiffuserConfig):
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
class LieGroupDiffuser(RiemannianManifoldDiffuser):
    def __init__(
        self,
        config: LieGroupDiffuserConfig,
        time_scheduler: DiffusionTimeScheduler,
        lie_group: LieGroup,
    ):
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
            riemannian_manifold=lie_group,
        )
        self.lie_group = lie_group
