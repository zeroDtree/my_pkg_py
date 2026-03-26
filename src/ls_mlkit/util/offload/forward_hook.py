import torch


class ForwardHookForDevice:
    def __init__(self):
        pass

    @staticmethod
    def get_align_device_pre_forward_hook(device="cuda", with_kwargs=False):
        """Ensure input tensors and module are on the same device before forward."""

        def hook(module: torch.nn.Module, args):
            p0 = next(module.parameters(), None)
            if device is not None:
                align_device = device
            elif p0 is not None:
                align_device = p0.device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def hook_with_kwargs(module: torch.nn.Module, args, kwargs):
            p0 = next(module.parameters(), None)
            if device is not None:
                align_device = device
            elif p0 is not None:
                align_device = p0.device
            else:
                align_device = "cuda"
            module.to(align_device)
            args = tuple(arg.to(align_device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {k: v.to(align_device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return args, kwargs

        if with_kwargs:
            return hook_with_kwargs
        else:
            return hook

    @staticmethod
    def get_forward_hook(pre: bool, device=None, with_kwargs=False):
        """
        Returns a pre- or post-forward hook that moves the module and inputs to
        ``device`` before the forward pass, and restores the module to its prior
        device after (post-forward hooks only).
        """
        origin_device = "cpu"
        if device is None:
            device = "cuda"

        def pre_hook(module: torch.nn.Module, args):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            return args

        def pre_hook_with_kwargs(module, args, kwargs):
            module.to(device)
            args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            kwargs = {n: v.to(device) if isinstance(v, torch.Tensor) else v for n, v in kwargs.items()}
            return args, kwargs

        def after_hook(module: torch.nn.Module, args, output):
            module.to(origin_device)
            return output

        def after_hook_with_kwargs(module, args, output):
            module.to(origin_device)
            return output

        if pre and with_kwargs:
            return pre_hook_with_kwargs
        elif pre:
            return pre_hook
        elif with_kwargs:
            return after_hook_with_kwargs
        else:
            return after_hook
