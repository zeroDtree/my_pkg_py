from abc import abstractmethod
from typing import Any, Optional

from ..util.base_class.base_gm_class import (
    BaseGenerativeModel,
    BaseGenerativeModelConfig,
)
from ..util.decorators import inherit_docstrings
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class BaseDiffuserConfig(BaseGenerativeModelConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ndim_micro_shape=ndim_micro_shape,
            n_discretization_steps=n_discretization_steps,
            n_inference_steps=n_inference_steps,
            *args,
            **kwargs,
        )


@inherit_docstrings
class BaseDiffuser(BaseGenerativeModel):
    """
    abstract method:
    """

    def __init__(
        self,
        config: BaseDiffuserConfig,
        time_scheduler: DiffusionTimeScheduler,
    ) -> None:
        r"""Initialize the BaseDiffuser

        Args:
            config (BaseDiffuserConfig): the config of the diffuser
            time_scheduler (DiffusionTimeScheduler): the time scheduler of the diffuser
        """
        super().__init__(config=config)
        self.config: BaseDiffuserConfig = config
        self.time_scheduler: DiffusionTimeScheduler = time_scheduler

    @abstractmethod
    def forward_process(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        """Diffuse a sample forward in time.

        Euclidean subclasses must implement the two-time contract::

            forward_process(x_start, t_a, t_b, mask, is_continuous_time=False, **kwargs)
                -> dict with at least {"x_t": Tensor}

        where ``x_start`` is a valid sample at noise level ``t_a`` and the
        result is the sample at noise level ``t_b`` (``t_b > t_a``).
        Other manifold backends may use a different signature.
        """
        return {}
