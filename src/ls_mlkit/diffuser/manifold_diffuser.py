r"""
Riemannian Manifold Diffuser
"""

from typing import Any

from ..util.decorators import inherit_docstrings
from ..util.manifold.riemannian_manifold import RiemannianManifold
from .base_diffuser import BaseDiffuser, BaseDiffuserConfig
from .time_scheduler import TimeScheduler


@inherit_docstrings
class RiemannianManifoldDiffuserConfig(BaseDiffuserConfig):
    """
    Riemannian Manifold Diffuser Config
    """

    def __init__(self, n_discretization_steps: int, ndim_micro_shape: int, *args: list[Any], **kwargs: dict[Any, Any]):
        super().__init__(
            n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape, *args, **kwargs
        )


@inherit_docstrings
class RiemannianManifoldDiffuser(BaseDiffuser):
    """
    Riemannian Manifold Diffuser
    """

    def __init__(
        self,
        config: RiemannianManifoldDiffuserConfig,
        time_scheduler: TimeScheduler,
        riemannian_manifold: RiemannianManifold,
    ):
        """Initialize the RiemannianManifoldDiffuser

        Args:
            config (RiemannianManifoldDiffuserConfig): the config of the RiemannianManifoldDiffuser
            time_scheduler (TimeScheduler): the time scheduler of the RiemannianManifoldDiffuser
            riemannian_manifold (RiemannianManifold): the Riemannian manifold of the RiemannianManifoldDiffuser
        """
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
        )
        self.riemannian_manifold = riemannian_manifold
