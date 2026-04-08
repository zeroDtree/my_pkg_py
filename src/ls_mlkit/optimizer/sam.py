from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.optim.optimizer import Optimizer


@dataclass
class SAMConfig:
    epsilon: float = field(default=1e-9)
    rho: float = field(default=0.05)
    adaptive: bool = field(default=False)


class SAM(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        model: torch.nn.Module,
        sam_config: SAMConfig = None,
        **kwargs,
    ):
        adaptive = sam_config.adaptive
        rho = sam_config.rho
        defaults = dict(adaptive=adaptive, rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        if isinstance(base_optimizer, type):
            base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.model = model
        self.sam_config = sam_config
        self.closure = None

    def set_closure(self, loss_fn, x, y, closure=None):
        if closure is not None:
            self.closure = closure
            return

        @torch.enable_grad()
        def _closure():
            self.zero_grad()
            o = self.model(x)
            loss = loss_fn(o, y)
            loss.backward()
            return loss.detach().clone().item()

        self.closure = _closure

    def step(self, closure=None):
        if closure is not None:
            self.closure = closure
        assert self.closure is not None, "closure is not set"
        loss = self.closure()
        grad_norm = self.get_gradient_norm(src="grad")
        theta = "theta"
        with torch.no_grad():
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + self.sam_config.epsilon)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p][theta] = p.data.clone()
                    grad = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad
                    p.data.add_(grad, alpha=scale)
        self.closure()
        x = torch.randn_like(p.data)
        x.norm()
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data = self.state[p][theta]
        self.base_optimizer.step()
        self.closure = None
        return loss

    @torch.no_grad()
    def get_gradient_norm(self, src: Literal["grad", "state", "weight"] = "grad", **kwargs):
        assert (
            src == "grad" or src == "state" or src == "weight"
        ), f"src must be in ['grad','state','weight'], {src} not"
        gradient_norm = 0.0
        if src == "grad":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    gradient_norm += torch.sum(grad * grad)
        elif src == "state":
            key = kwargs.get("key", None)
            assert key is not None, "src='state',please provide key of state"
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.state[p].get(key, None) is None:
                        continue
                    grad = self.state[p][key]
                    gradient_norm += torch.sum(grad * grad)
        elif src == "weight":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    gradient_norm += torch.sum(p.data * p.data)
        return torch.sqrt(gradient_norm)

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"SAM({self.base_optimizer.__class__.__name__})"


if __name__ == "__main__":
    pass
