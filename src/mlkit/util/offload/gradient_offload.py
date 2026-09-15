import torch.nn


class GradientOffloadHookContext:
    def __init__(self, model: torch.nn.Module, enable: bool, record_dict: dict, *args, **kwargs):
        """Offload gradients to CPU after each accumulation step.

        Args:
            model: The model whose gradients will be offloaded.
            enable: If False, this context is a no-op.
            record_dict: Dictionary that accumulates offloaded named gradients.
        """
        self.enable = enable
        if not enable:
            return
        self.model = model
        self.record_dict = record_dict
        self.offload_device = "cpu"
        self.handle_list: list = []

    def __enter__(self):
        if not self.enable:
            return self
        for name, param in self.model.named_parameters():
            handle = param.register_post_accumulate_grad_hook(
                self._make_offload_grad_hook(name, self.record_dict, self.offload_device)
            )
            self.handle_list.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return False
        for handle in self.handle_list:
            handle.remove()
        return False

    @staticmethod
    def _make_offload_grad_hook(name: str, record_dict: dict, offload_device: str):
        def offload_grad_hook(param):
            if param.grad is None:
                return
            grad = param.grad.to(offload_device)
            param.grad = None
            if name not in record_dict:
                record_dict[name] = grad
            else:
                acc = record_dict[name]
                if acc.dtype == grad.dtype and acc.device == grad.device:
                    acc.add_(grad)
                else:
                    record_dict[name] = acc + grad

        return offload_grad_hook
