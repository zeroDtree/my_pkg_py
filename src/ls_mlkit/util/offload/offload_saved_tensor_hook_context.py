import threading

import torch


def offload_condition(x: torch.Tensor) -> bool:
    """Return True if ``x`` is a dense CUDA tensor suitable for pack/offload.

    Requires the tensor to use a storage slice matching ``numel * element_size`` (typical
    for contiguous owned tensors). Views or tensors with shared/nonstandard storage may
    fail this check and are left unpacked so pack/unpack round-trips stay valid.
    """
    return x.device.type == "cuda" and x.numel() * x.element_size() == x.untyped_storage().size()


_thread_local = threading.local()


class OffloadSavedTensorHook:
    @staticmethod
    def _get_offload_device() -> str:
        return getattr(_thread_local, "offload_device", "cpu")

    @staticmethod
    def _set_offload_device(device: str) -> None:
        _thread_local.offload_device = device

    @staticmethod
    def unpack(packed):
        origin_device, x = packed
        return x.to(origin_device)

    @staticmethod
    def pack(x: torch.Tensor):
        if offload_condition(x):
            return x.device, x.detach().to(OffloadSavedTensorHook._get_offload_device())
        else:
            return x.device, x.detach()


class OffloadSavedTensorHookContext:
    def __init__(self):
        self.savedTensorOffloadContext = torch.autograd.graph.saved_tensors_hooks(
            pack_hook=OffloadSavedTensorHook.pack,
            unpack_hook=OffloadSavedTensorHook.unpack,
        )

    def __enter__(self):
        self.savedTensorOffloadContext.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.savedTensorOffloadContext.__exit__(exc_type, exc_val, exc_tb)
        return False
