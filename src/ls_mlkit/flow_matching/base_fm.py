from typing import Any

from ..util.base_class.base_gm_class import BaseGenerativeModel, BaseGenerativeModelConfig
from ..util.decorators import inherit_docstrings
from .time_scheduler import FlowMatchingTimeScheduler


@inherit_docstrings
class BaseFlowConfig(BaseGenerativeModelConfig):
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
class BaseFlow(BaseGenerativeModel):
    """
    abstract method: prior_sampling, compute_loss, step, sampling, inpainting
    """

    def __init__(
        self,
        config: BaseFlowConfig,
        time_scheduler: FlowMatchingTimeScheduler,
    ) -> None:
        super().__init__(config=config)
        self.config: BaseFlowConfig = config
        self.time_scheduler: FlowMatchingTimeScheduler = time_scheduler
