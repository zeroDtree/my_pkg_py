import torch


class KFA:
    known_modules = ["Linear"]
    eps = 1e-3

    def __init__(self, model):
        self.cache = {}
        self.handlers = []
        self.model = model

    def __enter__(self):
        self.register_save_hook(self.model)

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove_hook()

    def calculate_fisher_inverse_mult_V(
        cache: dict, a: torch.Tensor, g: torch.Tensor, V: torch.Tensor
    ):
        """
        V:(..., m, n)
        aaT:(..., m, m)
        ggT:(..., n, n)
        """
        a = a.unsqueeze(-2).transpose(-2, -1)
        aT = a.transpose(-2, -1)
        aaT = torch.matmul(a, aT)
        g = g.unsqueeze(-2).transpose(-2, -1)
        gT = g.transpose(-2, -1)
        ggT = torch.matmul(g, gT)
        EaaT = torch.mean(
            aaT,
            dim=tuple(range(len(aaT.shape) - 2)),
        )
        EggT = torch.mean(
            ggT,
            dim=tuple(range(len(ggT.shape) - 2)),
        )
        inv_EaaT = torch.linalg.inv(
            EaaT + KFA.eps * torch.eye(EaaT.shape[-1], device=EaaT.device)
        )
        # print(g)
        inv_EggT = torch.linalg.inv(
            EggT + KFA.eps * torch.eye(EggT.shape[-1], device=EggT.device)
        )
        result = torch.matmul(inv_EggT, V)
        result = torch.matmul(result, inv_EaaT)
        return result

    def get_save_hook_for_a(cache: dict, module_dot_path: str, name: str = "a"):
        def _save_hook_for_a(module: torch.nn.Module, args, kwargs):
            match module.__class__.__name__:
                case "Linear":
                    module: torch.nn.Linear
                    if cache.get(module.weight) is None:
                        cache[module.weight] = {}
                    cache[module.weight][name] = args[0]
                    if module.bias is not None:
                        if cache.get(module.bias) is None:
                            cache[module.bias] = {}
                        cache[module.bias][name] = torch.eye(
                            1, device=module.bias.device
                        )
                case _:
                    pass

        return _save_hook_for_a

    def get_save_hook_for_g(cache: dict, module_dot_path: str, name: str = "g"):

        def _save_hook_for_g(module, grad_input, grad_output):
            match module.__class__.__name__:
                case "Linear":
                    module: torch.nn.Linear
                    if cache.get(module.weight) is None:
                        cache[module.weight] = {}
                    cache[module.weight][name] = grad_output[0] * grad_output[0].shape[0]
                    if module.bias is not None:
                        if cache.get(module.bias) is None:
                            cache[module.bias] = {}
                        cache[module.bias][name] = grad_output[0] * grad_output[0].shape[0]
                case _:
                    pass

        return _save_hook_for_g

    def register_save_hook(self, model):
        def _register_save_hook(module: torch.nn.Module, prefix=""):
            if module.__class__.__name__ in self.known_modules:
                handler = module.register_forward_pre_hook(
                    KFA.get_save_hook_for_a(
                        cache=self.cache,
                        module_dot_path=prefix,
                    ),
                    with_kwargs=True,
                )
                self.handlers.append(handler)
                handler = module.register_full_backward_hook(
                    KFA.get_save_hook_for_g(
                        cache=self.cache,
                        module_dot_path=prefix,
                    ),
                )
                self.handlers.append(handler)
            if len(list(module.children())) <= 0:
                return
            for name, submodule in module.named_children():
                new_prefix = prefix + "." + name if prefix != "" else name
                _register_save_hook(module=submodule, prefix=new_prefix)

        _register_save_hook(module=model, prefix="")

    def remove_hook(self):
        for handler in self.handlers:
            handler.remove()
