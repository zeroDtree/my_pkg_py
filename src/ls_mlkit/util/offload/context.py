from contextlib import ExitStack

import torch.nn

from .gradient_offload import GradientOffloadHookContext
from .model_offload import ModelOffloadHookContext


class OffloadContext:
    def __init__(
        self,
        model: torch.nn.Module,
        named_grads: dict,
        num_block=2,
        no_split_module_classes=None,
        enable_gradient_offload=True,
        enable_model_offload=True,
    ):
        self.modelOffloadHookContext = ModelOffloadHookContext(
            model=model,
            no_split_module_classes=no_split_module_classes,
            num_block=num_block,
            enable=enable_model_offload,
            device="cuda",
            strategy="block",
        )
        self.gradientOffloadHookContext = GradientOffloadHookContext(
            model=model,
            enable=enable_gradient_offload,
            record_dict=named_grads,
        )

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.enter_context(self.modelOffloadHookContext)
        self._stack.enter_context(self.gradientOffloadHookContext)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stack.__exit__(exc_type, exc_val, exc_tb)
