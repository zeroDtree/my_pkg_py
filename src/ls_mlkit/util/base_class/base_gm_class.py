from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, cast

from torch import Tensor

from ..decorators import inherit_docstrings
from ..hook.base_hook import Hook, HookHandler, HookManager
from .base_loss_class import BaseLoss, BaseLossConfig


class GMHookStageType(Enum):
    PRE_UPDATE_IN_STEP_FN = "pre_update_in_step_fn"
    POST_UPDATE_IN_STEP_FN = "post_update_in_step_fn"
    PRE_COMPUTE_LOSS = "pre_compute_loss"
    POST_COMPUTE_LOSS = "post_compute_loss"
    POST_SAMPLING_TIME_STEP = "post_sampling_time_step"
    POST_GET_MACRO_SHAPE = "get_macro_shape"


class GMHookHandler(HookHandler[GMHookStageType]):
    pass


class GMHook(Hook[GMHookStageType]):
    pass


class GMHookManager(HookManager[GMHookStageType]):
    pass


@inherit_docstrings
class BaseGenerativeModelConfig(BaseLossConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        n_discretization_steps: int,
        n_inference_steps: Optional[int] = None,
        **kwargs: dict[Any, Any],
    ):
        super().__init__(ndim_micro_shape=ndim_micro_shape, **kwargs)
        self.n_discretization_steps: int = n_discretization_steps
        if n_inference_steps is not None:
            self.n_inference_steps: int = n_inference_steps
        else:
            self.n_inference_steps: int = n_discretization_steps


@inherit_docstrings
class BaseGenerativeModel(BaseLoss):
    """
    abstract method: compute_loss, step, sampling, inpainting
    """

    def __init__(self, config: BaseGenerativeModelConfig):
        super().__init__(config=config)
        self.config: BaseGenerativeModelConfig = config
        self.hook_manager = GMHookManager()

    @abstractmethod
    def prior_sampling(self, shape: tuple[int, ...]) -> Tensor: ...

    @abstractmethod
    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Optional[Tensor] = None,
        **kwargs: dict[Any, Any],
    ) -> dict:
        """_summary_

        Args:
            x_t (``Tensor``): _description_
            t (``Tensor``): _description_
            padding_mask (``Tensor``, *optional*): _description_. Defaults to None.

        Returns:
            ``dict``: A dictionary that must contain the key "x"
        """

    @abstractmethod
    def sampling(
        self,
        shape,
        device,
        x_init_posterior=None,
        return_all=False,
        sampling_condition=None,
        sapmling_condition_key="sapmling_condition",
        **kwargs,
    ) -> dict: ...

    @abstractmethod
    def inpainting(
        self,
        x,
        padding_mask,
        inpainting_mask,
        device,
        x_init_posterior: Optional[Tensor] = None,
        inpainting_mask_key: str = "inpainting_mask",
        sapmling_condition_key: Optional[str] = "sapmling_condition",
        return_all: bool = False,
        sampling_condition: Optional[Any] = None,
        **kwargs,
    ) -> dict: ...

    def forward(self, **batch) -> dict:
        r"""Forward function, input batch of data and return the dictionary containing the loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict``: a dictionary that must contain the key "loss"
        """
        result = self.compute_loss(**batch)
        hook_result = self.hook_manager.run_hooks(stage=GMHookStageType.POST_COMPUTE_LOSS, tgt_key_name=None, **result)
        if hook_result is not None:
            assert isinstance(hook_result, (dict, Tensor))
            result = hook_result
        return result

    def register_post_compute_loss_hook(
        self,
        name: str,
        fn: Callable[..., Any],
        priority: int = 0,
        enabled: bool = True,
    ) -> GMHookHandler:
        r"""Register a hook to be called after loss computation

        Args:
            name (``str``): the name of the hook
            fn (``Callable[..., Any]``): the function to be called
            priority (``int``, optional): the priority of the hook. Defaults to 0.
            enabled (``bool``, optional): whether the hook is enabled. Defaults to True.
        """
        hook = Hook(
            name=name,
            stage=GMHookStageType.POST_COMPUTE_LOSS,
            fn=fn,
            priority=priority,
            enabled=enabled,
        )
        handler = self.hook_manager.register_hook(hook)
        handler = cast(GMHookHandler, handler)
        return handler

    def register_hooks(self, hooks: list[GMHook]) -> list[GMHookHandler]:
        handler_list = []
        for hook in hooks:
            handler = self.hook_manager.register_hook(hook)
            handler = cast(GMHookHandler, handler)
            handler_list.append(handler)
        return handler_list

    def register_hook(self, hook: GMHook) -> GMHookHandler:
        handler = self.hook_manager.register_hook(hook)
        handler = cast(GMHookHandler, handler)
        return handler
