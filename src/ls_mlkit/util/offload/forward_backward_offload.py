import torch

from .forward_hook import ForwardHookForDevice
from .offload_saved_tensor_hook_context import OffloadSavedTensorHook
from .utils import get_module_list, get_partition_block


class ForwardBackwardOffloadHookContext(ForwardHookForDevice):
    def __init__(
        self,
        model,
        device="cuda",
        no_split_module_classes=None,
        enable=True,
        num_block: int = 2,
        strategy="block",
    ):
        """Offload model weights to CPU between forward/backward sub-blocks.

        Args:
            model: The model to which hooks will be applied.
            device: The compute device (e.g. "cuda").
            no_split_module_classes: Module class names that should not be split further.
            enable: If False, this context is a no-op.
            num_block: Number of blocks for the "block" strategy.
            strategy: Only ``'block'`` is implemented: partition leaf boundary modules into
                ``num_block`` groups.
        """
        super().__init__()
        self.enable = enable
        if not enable:
            return
        if strategy != "block":
            raise ValueError(f"Unsupported strategy {strategy!r}; only 'block' is implemented.")
        self.strategy = strategy
        self.num_block = num_block
        self.device = device
        self.model = model
        self.handle_list: list = []
        self.module_list = get_module_list(model, no_split_module_classes=no_split_module_classes)
        self.module_info = get_partition_block(self.module_list, self.num_block)

    def __enter__(self):
        if not self.enable:
            return self
        self.register_hook_by_block(self.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return False
        for handle in self.handle_list:
            handle.remove()
        return False

    def register_hook_by_block(self, module: torch.nn.Module, parent_name=""):
        if parent_name and parent_name in self.module_list:
            handle = module.register_forward_pre_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=True, device=self.device),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_forward_hook(
                hook=self.get_forward_hook_by_block(info=self.module_info[parent_name], pre=False, device=self.device),
                with_kwargs=True,
            )
            self.handle_list.append(handle)
            handle = module.register_full_backward_pre_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=True, device=self.device)
            )
            self.handle_list.append(handle)
            handle = module.register_full_backward_hook(
                hook=self.get_backward_hook_by_block(info=self.module_info[parent_name], pre=False, device=self.device)
            )
            self.handle_list.append(handle)
            return

        for name, sub_module in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            self.register_hook_by_block(sub_module, full_name)

    @staticmethod
    def get_forward_hook_by_block(info: dict, pre=True, device="cuda"):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        last_block_flag = info["last_block_flag"]
        first_module_flag = info["first_module_flag"]

        def pre_hook(module, args, kwargs):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            # Steer saved-tensor offloading: non-last blocks offload activations to CPU.
            if first_module_flag:
                OffloadSavedTensorHook._set_offload_device(offload_device if not last_block_flag else device)
            return args, kwargs

        def after_hook(module, args, kwargs, output):
            module.to(offload_device if not last_block_flag else device)
            return output

        if pre:
            return pre_hook
        else:
            return after_hook

    @staticmethod
    def get_backward_hook_by_block(info: dict, pre=True, device="cuda"):
        if device is None:
            device = "cuda"
        offload_device = "cpu"
        first_block_flag = info["first_block_flag"]

        def pre_hook(module, grad_output):
            module.to(device)
            return grad_output

        def after_hook(module, grad_input, grad_output):
            if not first_block_flag:
                module.to(offload_device)
            return grad_input

        if pre:
            return pre_hook
        else:
            return after_hook
