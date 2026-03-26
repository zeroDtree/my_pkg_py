from contextlib import ExitStack

from .forward_backward_offload import ForwardBackwardOffloadHookContext
from .offload_saved_tensor_hook_context import OffloadSavedTensorHookContext


class ModelOffloadHookContext:
    def __init__(
        self,
        model,
        no_split_module_classes=None,
        num_block: int = 2,
        enable=True,
        device="cuda",
        strategy="block",
    ):
        """Combine forward/backward offloading with saved-tensor offloading.

        Args:
            model: The model to which hooks will be applied.
            no_split_module_classes: Module class names that should not be split further.
            num_block: Number of blocks for the "block" strategy.
            enable: If False, this context is a no-op.
            device: The compute device (e.g. "cuda").
            strategy: Only ``'block'`` is supported (see ``ForwardBackwardOffloadHookContext``).
        """
        self.enable = enable
        if not enable:
            return
        self.forwardBackwardOffloadHookContext = ForwardBackwardOffloadHookContext(
            model=model,
            device=device,
            no_split_module_classes=no_split_module_classes,
            enable=True,
            num_block=num_block,
            strategy=strategy,
        )
        self.savedTensorOffloadContext = OffloadSavedTensorHookContext()

    def __enter__(self):
        self._stack = ExitStack()
        if self.enable:
            self._stack.enter_context(self.forwardBackwardOffloadHookContext)
            self._stack.enter_context(self.savedTensorOffloadContext)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stack.__exit__(exc_type, exc_val, exc_tb)
