r"""
Lie Group Diffuser
"""

from typing import Any

from ..util.decorators import inherit_docstrings
from ..util.manifold.lie_group import LieGroup
from .manifold_diffuser import RiemannianManifoldDiffuser, RiemannianManifoldDiffuserConfig
from .time_scheduler import TimeScheduler


@inherit_docstrings
class LieGroupDiffuserConfig(RiemannianManifoldDiffuserConfig):
    """
    Lie Group Diffuser Config
    """

    def __init__(self, n_discretization_steps: int, ndim_micro_shape: int, *args: list[Any], **kwargs: dict[Any, Any]):
        super().__init__(
            n_discretization_steps=n_discretization_steps, ndim_micro_shape=ndim_micro_shape, *args, **kwargs
        )


@inherit_docstrings
class LieGroupDiffuser(RiemannianManifoldDiffuser):
    """
    Riemannian Manifold Diffuser
    """

    def __init__(
        self,
        config: LieGroupDiffuserConfig,
        time_scheduler: TimeScheduler,
        lie_group: LieGroup,
    ):
        """Initialize the LieGroupDiffuser

        Args:
            config (LieGroupDiffuserConfig): the config of the LieGroupDiffuser
            time_scheduler (TimeScheduler): the time scheduler of the LieGroupDiffuser
            lie_group (LieGroup): the Lie group of the LieGroupDiffuser
        """
        super().__init__(
            config=config,
            time_scheduler=time_scheduler,
            riemannian_manifold=lie_group,
        )
        self.lie_group = lie_group
