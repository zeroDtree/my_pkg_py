import torch
from torch.optim import Optimizer
from typing import Literal
from .kfa import KFA
from dataclasses import dataclass, field
import math


@dataclass
class UserConfig:
    xi: float = field(default=1e-4)
    alpha: float = field(default=0.9)
    rho: float = field(default=0.1)
    rho_cov: float = field(default=0.1)


class KFAOptimizer(Optimizer):
    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        model: torch.nn.Module,
        user_config: UserConfig,
        kfa: KFA,
        **kwargs,
    ):
        rho = user_config.rho
        rho_cov = user_config.rho_cov
        defaults = dict(rho=rho, rho_cov=rho_cov, **kwargs)
        super(KFAOptimizer, self).__init__(params, defaults)
        if isinstance(base_optimizer, type):
            base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.model = model
        self.user_config = user_config
        self.kfa = kfa

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
        with self.kfa:
            loss = self.closure()
        self.save_params("theta")
        self.save_grad("g1")
        self.perturb(name="grad")
        self.closure()
        self.save_grad("g2")
        self.save_moving_average_and_save_d(name="d", alpha=self.user_config.alpha)
        self.calculate_fisher_inverse_mult(tgt_key="inv_Fd", src_key="d")
        self.calculate_fisher_inverse_mult(tgt_key="inv_Fg1", src_key="g1")
        self.calculate_C_inverse_mult_d(tgt_key="inv_Cd")
        self.epsilon_perturb()
        self.closure()
        self.back_to("theta")
        self.base_optimizer.step()
        self.closure = None
        return loss

    @torch.no_grad()
    def epsilon_perturb(self):
        scale = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale += torch.sum(self.state[p]["d"] * self.state[p]["inv_Cd"])
        scale = 1 / torch.sqrt(scale)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["epsilon"] = scale * self.state[p]["inv_Cd"]
        self.back_to("theta")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = p.data + group["rho_cov"] * self.state[p]["epsilon"]

    @torch.no_grad()
    def calculate_C_inverse_mult_d(self, tgt_key: str = "inv_Cd"):
        numerator, denominator = 0.0, 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                numerator += torch.sum(self.state[p]["g1"] * self.state[p]["inv_Fd"])
                denominator += torch.sum(self.state[p]["g1"] * self.state[p]["inv_Fg1"])
        print(denominator)
        # denominator = 1 - torch.exp(-denominator)
        denominator = 1 - denominator
        # denominator = 1
        scale = numerator / denominator
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p][tgt_key] = (
                    self.state[p]["inv_Fd"] + scale * self.state[p]["inv_Fg1"]
                )

    @torch.no_grad()
    def calculate_fisher_inverse_mult(self, tgt_key: str, src_key: str):
        def _calculate_fisher_inverse_mult(module: torch.nn.Module, prefix: str):
            if module.__class__.__name__ in self.kfa.known_modules:
                match module.__class__.__name__:
                    case "Linear":
                        """
                        $\F^{-1}d$
                        """
                        if not module.weight.requires_grad:
                            return
                        self.state[module.weight][tgt_key] = (
                            KFA.calculate_fisher_inverse_mult_V(
                                cache=self.kfa.cache,
                                a=self.kfa.cache[module.weight]["a"],
                                g=self.kfa.cache[module.weight]["g"],
                                V=self.state[module.weight][src_key],
                            )
                        )
                        if module.bias is not None:
                            self.state[module.bias][tgt_key] = (
                                KFA.calculate_fisher_inverse_mult_V(
                                    cache=self.kfa.cache,
                                    a=self.kfa.cache[module.bias]["a"],
                                    g=self.kfa.cache[module.bias]["g"],
                                    V=self.state[module.bias][src_key]
                                    .unsqueeze(-2)
                                    .transpose(-2, -1),
                                )
                                .transpose(-2, -1)
                                .squeeze(-2)
                            )
                    case _:
                        raise ValueError(f"unknown module: {module.__class__.__name__}")
                return
            for name, sub_module in module.named_children():
                prefix = f"{prefix}.{name}" if prefix else name
                _calculate_fisher_inverse_mult(sub_module, prefix)

        _calculate_fisher_inverse_mult(self.model, "")

    @torch.no_grad()
    def back_to(self, name):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p][name]

    @torch.no_grad()
    def perturb(self, name: Literal["grad", "state"] = "grad", **kwargs):
        if name == "grad":
            grad_norm = self.get_something_norm(name, **kwargs) + self.user_config.xi
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if name == "grad":
                        grad = p.grad
                    elif name == "state":
                        key = kwargs.get("key", None)
                        assert (
                            key is not None
                        ), "src='state',please provide key of state"
                        grad = self.state[p][key]
                    p.data = p.data + group["rho"] * grad / grad_norm

    @torch.no_grad()
    def save_moving_average_and_save_d(self, name: str, alpha: float = 0.9):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p]["g2"] - self.state[p]["g1"]
                if self.state[p].get(name, None) is None:
                    self.state[p][name] = torch.zeros_like(m)
                else:
                    self.state[p][name] = alpha * self.state[p][name] + (1 - alpha) * m
                self.state[p]["d"] = m - self.state[p][name]

    @torch.no_grad()
    def save_grad(self, name: str):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p][name] = p.grad.detach().clone()

    @torch.no_grad()
    def save_params(self, name: str):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p][name] = p.data.detach().clone()

    @torch.no_grad()
    def get_something_norm(
        self, something_name: Literal["grad", "state", "weight"] = "grad", **kwargs
    ):
        assert (
            something_name == "grad"
            or something_name == "state"
            or something_name == "weight"
        ), f"something_name must be in ['grad','state','weight'], {something_name} not"
        something_norm = 0.0
        if something_name == "grad":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = (
                        torch.abs(p) if group.get("adaptive", None) else 1.0
                    ) * p.grad
                    something_norm += torch.sum(grad * grad).item()
        elif something_name == "state":
            key = kwargs.get("key", None)
            assert key is not None, "src='state',please provide key of state"
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if self.state[p].get(key, None) is None:
                        continue
                    grad = self.state[p][key]
                    something_norm += torch.sum(grad * grad).item()
        elif something_name == "weight":
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    something_norm += torch.sum(p.data * p.data).item()
        return math.sqrt(something_norm)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"KFAOptimizer({self.base_optimizer.__class__.__name__})"
