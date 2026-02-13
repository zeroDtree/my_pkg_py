from abc import abstractmethod
from typing import Any

from torch import Tensor

from ..util.base_class.base_gm_class import BaseGenerativeModel, BaseGenerativeModelConfig
from ..util.decorators import inherit_docstrings
from .time_scheduler import DiffusionTimeScheduler


@inherit_docstrings
class BaseDiffuserConfig(BaseGenerativeModelConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: int = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
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
            config (``BaseDiffuserConfig``): the config of the diffuser
            time_scheduler (``DiffusionTimeScheduler``): the time scheduler of the diffuser
        """
        super().__init__(config=config)
        self.config: BaseDiffuserConfig = config
        self.time_scheduler: DiffusionTimeScheduler = time_scheduler

    @abstractmethod
    def forward_process(
        self,
        x_0: Tensor,
        t_a: Tensor,
        t_b: Tensor,
        mask: Tensor,
        is_continuous_time: bool = True,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ) -> dict:
        assert (t_b >= t_a).all()
