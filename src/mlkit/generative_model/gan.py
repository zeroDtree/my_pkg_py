"""Classic GAN with non-saturating generator loss and BCE-with-logits objective."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits

from ..util.base_class.base_loss_class import BaseLoss, BaseLossConfig
from ..util.decorators import inherit_docstrings
from ..util.ema import EMA


@inherit_docstrings
class GANConfig(BaseLossConfig):
    def __init__(
        self,
        ndim_micro_shape: int = 1,
        latent_dim: int = 32,
        n_critic: int = 1,
        r1_gamma: float = 0.0,
        ema_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(ndim_micro_shape=ndim_micro_shape, **kwargs)
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.r1_gamma = r1_gamma
        self.ema_decay = ema_decay


@inherit_docstrings
class GAN(BaseLoss):
    """Non-saturating GAN with injected generator and discriminator networks.

    The generator maps ``batch["z"]`` to ``{"x": Tensor}``.
    The discriminator maps ``batch["gt_data"]`` to ``{"logits": Tensor}`` with
    shape ``(B,)`` (raw logits, no sigmoid).

    ``forward`` alternates discriminator and generator sub-losses so a single
    optimizer / pipeline step updates only one network at a time. When
    ``ema_decay > 0``, an EMA of the generator is updated only on generator steps.
    """

    def __init__(
        self,
        config: GANConfig,
        generator: Module,
        discriminator: Module,
    ) -> None:
        super().__init__(config=config)
        self.config: GANConfig = config
        self.generator = generator
        self.discriminator = discriminator
        self.ema: EMA | None = EMA(self.generator, decay=config.ema_decay) if config.ema_decay > 0 else None
        self._step_count = 0

    def prior_sampling(self, shape: tuple[int, ...]) -> Tensor:
        return torch.randn(shape)

    def _latent_shape_from_output_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        macro_shape = shape[: len(shape) - self.config.ndim_micro_shape]
        return (*macro_shape, self.config.latent_dim)

    def generate(self, z: Tensor, **batch: Any) -> Tensor:
        batch = {**batch, "z": z}
        return self.generator(**batch)["x"]

    def discriminator_logits(self, x: Tensor, **batch: Any) -> Tensor:
        batch = {**batch, "gt_data": x}
        return self.discriminator(**batch)["logits"].reshape(-1)

    @contextmanager
    def _discriminator_frozen(self) -> Iterator[None]:
        requires_grad = [param.requires_grad for param in self.discriminator.parameters()]
        try:
            for param in self.discriminator.parameters():
                param.requires_grad_(False)
            yield
        finally:
            for param, flag in zip(self.discriminator.parameters(), requires_grad, strict=True):
                param.requires_grad_(flag)

    def _r1_penalty(self, real_logits: Tensor, x: Tensor) -> Tensor:
        (grad,) = torch.autograd.grad(
            real_logits.sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )
        return grad.flatten(1).pow(2).sum(dim=-1).mean()

    @torch.no_grad()
    def sampling(self, shape: tuple[int, ...], device: torch.device | str, **kwargs: Any) -> dict[str, Tensor]:
        z = self.prior_sampling(self._latent_shape_from_output_shape(shape)).to(device)
        fake = self.generate(z, **kwargs)
        return {"x": fake, "z": z}

    def compute_discriminator_loss(self, **batch: Any) -> dict[str, Any]:
        x: Tensor = batch["gt_data"]
        batch.setdefault("padding_mask", torch.ones_like(x))

        z = self.prior_sampling(self._latent_shape_from_output_shape(tuple(x.shape))).to(x.device)
        with torch.no_grad():
            fake = self.generate(z, **batch)

        x_for_d = x.detach().requires_grad_(self.config.r1_gamma > 0)
        real_logits = self.discriminator_logits(x_for_d, **batch)
        fake_logits = self.discriminator_logits(fake, **batch)

        d_loss_real = binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        d_loss_fake = binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        d_loss = d_loss_real + d_loss_fake

        r1 = torch.zeros((), device=x.device, dtype=d_loss.dtype)
        if self.config.r1_gamma > 0:
            r1 = self._r1_penalty(real_logits, x_for_d)
            d_loss = d_loss + 0.5 * self.config.r1_gamma * r1

        return {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
            "r1": r1.detach(),
            "gt_data": x,
            "fake": fake,
            "z": z,
            "real_logits": real_logits.detach(),
            "fake_logits": fake_logits.detach(),
            "padding_mask": batch["padding_mask"],
            "batch": batch,
        }

    def compute_generator_loss(self, **batch: Any) -> dict[str, Any]:
        x: Tensor = batch["gt_data"]
        batch.setdefault("padding_mask", torch.ones_like(x))

        z = self.prior_sampling(self._latent_shape_from_output_shape(tuple(x.shape))).to(x.device)
        fake = self.generate(z, **batch)

        with self._discriminator_frozen():
            fake_logits = self.discriminator_logits(fake, **batch)
            g_loss = binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        return {
            "g_loss": g_loss,
            "gt_data": x,
            "fake": fake,
            "z": z,
            "fake_logits": fake_logits.detach(),
            "padding_mask": batch["padding_mask"],
            "batch": batch,
        }

    def compute_loss(self, **batch: Any) -> dict[str, Any]:
        """Joint D+G loss for smoke tests / single-optimizer callers (no EMA update)."""
        d_out = self.compute_discriminator_loss(**batch)
        g_out = self.compute_generator_loss(**batch)
        loss = d_out["d_loss"] + g_out["g_loss"]
        return {
            "loss": loss,
            "d_loss": d_out["d_loss"],
            "g_loss": g_out["g_loss"],
            "d_loss_real": d_out["d_loss_real"],
            "d_loss_fake": d_out["d_loss_fake"],
            "r1": d_out["r1"],
            "gt_data": d_out["gt_data"],
            "fake": g_out["fake"],
            "z": g_out["z"],
            "real_logits": d_out["real_logits"],
            "fake_logits": g_out["fake_logits"],
            "padding_mask": d_out["padding_mask"],
            "batch": batch,
        }

    def forward(self, **batch: Any) -> dict[str, Any]:
        cycle_len = self.config.n_critic + 1
        is_generator_step = (self._step_count % cycle_len) == self.config.n_critic
        self._step_count += 1
        if is_generator_step:
            out = self.compute_generator_loss(**batch)
            if self.ema is not None:
                self.ema.update(self.generator)
            return {"loss": out["g_loss"], **out}
        out = self.compute_discriminator_loss(**batch)
        return {"loss": out["d_loss"], **out}


if __name__ == "__main__":
    from torch import nn

    batch_size = 8
    data_dim = 16
    latent_dim = 4

    class _Generator(Module):
        def __init__(self, latent_dim: int, data_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ELU(),
                nn.Linear(32, data_dim),
            )

        def forward(self, **batch: Any) -> dict[str, Tensor]:
            return {"x": self.net(batch["z"])}

    class _Discriminator(Module):
        def __init__(self, data_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, 32),
                nn.ELU(),
                nn.Linear(32, 1),
            )

        def forward(self, **batch: Any) -> dict[str, Tensor]:
            return {"logits": self.net(batch["gt_data"]).squeeze(-1)}

    config = GANConfig(ndim_micro_shape=1, latent_dim=latent_dim, n_critic=1, r1_gamma=1.0, ema_decay=0.999)
    gan = GAN(
        config=config,
        generator=_Generator(latent_dim, data_dim),
        discriminator=_Discriminator(data_dim),
    )

    x = torch.randn(batch_size, data_dim)
    batch = {"gt_data": x, "padding_mask": torch.ones_like(x)}
    result = gan.compute_loss(**batch)
    result["loss"].backward()

    d_step = gan(**batch)
    g_step = gan(**batch)
    assert "d_loss" in d_step and "g_loss" in g_step
    assert gan.ema is not None

    samples = gan.sampling((4, data_dim), device=x.device)
    print(
        f"loss={result['loss'].item():.4f}, "
        f"d_loss={result['d_loss'].item():.4f}, "
        f"g_loss={result['g_loss'].item():.4f}, "
        f"r1={result['r1'].item():.4f}"
    )
    print(f"sample shape={samples['x'].shape}")
    print(f"alt steps: d={d_step['loss'].item():.4f}, g={g_step['loss'].item():.4f}")
