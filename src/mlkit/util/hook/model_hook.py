from enum import Enum

from .base_hook import Hook, HookHandler, HookManager


class ModelHookStageType(Enum):
    PRE_COMPUTE_LOSS = "pre_compute_loss"
    POST_COMPUTE_LOSS = "post_compute_loss"


class ModelHookHandler(HookHandler[ModelHookStageType]):
    pass


class ModelHook(Hook[ModelHookStageType]):
    pass


class ModelHookManager(HookManager[ModelHookStageType]):
    pass
