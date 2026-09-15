"""Score-based Energy-Based Model with DSM training and Langevin sampling."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_loss_class import BaseLoss, BaseLossConfig
from ..util.decorators import inherit_docstrings


@inherit_docstrings
class EnergyBasedModelConfig(BaseLossConfig):
    def __init__(
        self,
        ndim_micro_shape: int = 1,
        sigma: float = 0.5,
        langevin_steps: int = 200,
        langevin_step_size: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(ndim_micro_shape=ndim_micro_shape, **kwargs)
        self.sigma = sigma
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size


@inherit_docstrings
class EnergyBasedModel(BaseLoss):
    """Energy network + DSM objective + Langevin Dynamics sampling.

    The energy net maps a batch dict (at least ``gt_data``) to
    ``{"energy": Tensor}`` with shape ``(B,)``. The score is obtained by
    automatic differentiation: ``s_phi(x) = -grad_x E_phi(x)``.
    """

    def __init__(self, config: EnergyBasedModelConfig, energy_net: Module) -> None:
        super().__init__(config=config)
        self.config: EnergyBasedModelConfig = config
        self.energy_net = energy_net

    def energy(self, **batch: Any) -> Tensor:
        out = self.energy_net(**batch)
        energy = out["energy"]
        return energy.reshape(-1)

    def score(self, x: Tensor, *, create_graph: bool = False, **batch_extra: Any) -> Tensor:
        """Return ``s_phi(x) = -nabla_x E_phi(x)``."""
        x = x.detach().requires_grad_(True)
        energy = self.energy(gt_data=x, **batch_extra)
        (grad,) = torch.autograd.grad(
            energy.sum(),
            x,
            create_graph=create_graph,
            retain_graph=create_graph,
        )
        return -grad

    def prior_sampling(self, shape: tuple[int, ...]) -> Tensor:
        return torch.randn(shape)

    def compute_loss(self, **batch: Any) -> dict[str, Any]:
        x: Tensor = batch["gt_data"]
        batch.setdefault("padding_mask", torch.ones_like(x))
        padding_mask = batch["padding_mask"]

        sigma = self.config.sigma
        eps = torch.randn_like(x)
        x_tilde = x + sigma * eps
        target = (x - x_tilde) / (sigma**2)

        score_pred = self.score(x_tilde, create_graph=True)
        # Mean over batch; flatten micro dims into the squared norm.
        per_sample = 0.5 * (score_pred - target).flatten(1).pow(2).sum(dim=-1)
        loss = per_sample.mean()

        return {
            "loss": loss,
            "dsm_loss": loss,
            "gt_data": x,
            "x_tilde": x_tilde,
            "score_pred": score_pred.detach(),
            "score_target": target,
            "padding_mask": padding_mask,
            "batch": batch,
        }

    def forward(self, **batch: Any) -> dict[str, Any]:
        return self.compute_loss(**batch)

    @torch.no_grad()
    def sampling(
        self,
        shape: tuple[int, ...],
        device: torch.device | str,
        x_init: Tensor | None = None,
        return_all: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Langevin Dynamics with the learned score field."""
        eta = self.config.langevin_step_size
        n_steps = self.config.langevin_steps
        noise_scale = (2.0 * eta) ** 0.5

        x = x_init.to(device) if x_init is not None else self.prior_sampling(shape).to(device)
        x_list: list[Tensor] | None = [x.detach().cpu()] if return_all else None

        for _ in range(n_steps):
            with torch.enable_grad():
                s = self.score(x, create_graph=False)
            x = x + eta * s + noise_scale * torch.randn_like(x)
            if x_list is not None:
                x_list.append(x.detach().cpu())

        result: dict[str, Any] = {"x": x}
        if x_list is not None:
            result["x_list"] = x_list
        return result


if __name__ == "__main__":
    from torch import nn

    batch_size = 8
    data_dim = 16

    class _EnergyNet(Module):
        def __init__(self, data_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, 32),
                nn.ELU(),
                nn.Linear(32, 32),
                nn.ELU(),
                nn.Linear(32, 1),
            )

        def forward(self, **batch: Any) -> dict[str, Tensor]:
            return {"energy": self.net(batch["gt_data"]).squeeze(-1)}

    config = EnergyBasedModelConfig(ndim_micro_shape=1, sigma=0.5, langevin_steps=10)
    ebm = EnergyBasedModel(config=config, energy_net=_EnergyNet(data_dim))

    x = torch.randn(batch_size, data_dim)
    batch = {"gt_data": x, "padding_mask": torch.ones_like(x)}
    result = ebm.compute_loss(**batch)
    result["loss"].backward()

    samples = ebm.sampling((4, data_dim), device=x.device)
    print(f"loss={result['loss'].item():.4f}")
    print(f"sample shape={samples['x'].shape}")
