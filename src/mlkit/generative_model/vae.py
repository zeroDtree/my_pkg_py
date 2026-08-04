"""Standard diagonal-Gaussian Variational Autoencoder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from ..util.base_class.base_loss_class import BaseLoss, BaseLossConfig
from ..util.context.temp_remove import TemporaryKeyRemover
from ..util.decorators import inherit_docstrings

LOGVAR_CLAMP_MIN = -20.0
LOGVAR_CLAMP_MAX = 20.0


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """Closed-form KL(q(z|x) || N(0, I)) for diagonal Gaussian q."""
    logvar = logvar.clamp(min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl_per_sample.mean()


@inherit_docstrings
class GaussianVAEConfig(BaseLossConfig):
    def __init__(
        self,
        ndim_micro_shape: int = 1,
        latent_dim: int = 32,
        decoder_sigma: float = 1.0,
        kl_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(ndim_micro_shape=ndim_micro_shape, **kwargs)
        self.latent_dim = latent_dim
        self.decoder_sigma = decoder_sigma
        self.kl_weight = kl_weight


@inherit_docstrings
class GaussianVAE(BaseLoss):
    def __init__(
        self,
        config: GaussianVAEConfig,
        encoder: Module,
        decoder: Module,
        recon_loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> None:
        super().__init__(config=config)
        self.config: GaussianVAEConfig = config
        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_fn = recon_loss_fn

    def encode(self, **batch) -> dict[str, Tensor]:
        return self.encoder(**batch)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = logvar.clamp(min=LOGVAR_CLAMP_MIN, max=LOGVAR_CLAMP_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, **batch) -> Tensor:
        return self.decoder(**batch)["x"]

    def prior_sampling(self, shape: tuple[int, ...]) -> Tensor:
        return torch.randn(shape)

    def _latent_shape_from_output_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        macro_shape = shape[: len(shape) - self.config.ndim_micro_shape]
        return (*macro_shape, self.config.latent_dim)

    @torch.no_grad()
    def sampling(self, shape: tuple[int, ...], device: torch.device | str, **kwargs: Any) -> dict[str, Tensor]:
        z = self.prior_sampling(self._latent_shape_from_output_shape(shape)).to(device)
        batch = {"z": z, **kwargs}
        return {"x": self.decode(**batch), "z": z}

    def compute_loss(self, **batch) -> dict[str, Any]:
        x = batch["gt_data"]
        batch.setdefault("padding_mask", torch.ones_like(x))
        padding_mask = batch["padding_mask"]

        encoded = self.encode(**batch)
        mu, logvar = encoded["mu"], encoded["logvar"]
        batch["mu"] = mu
        batch["logvar"] = logvar
        z = self.reparameterize(mu, logvar)
        batch["z"] = z

        with TemporaryKeyRemover(mapping=batch, keys=["gt_data"]):
            x_hat = self.decode(**batch)

        recon = self.recon_loss_fn(x_hat, x, padding_mask)
        recon = recon / (2 * self.config.decoder_sigma**2)
        kl = kl_standard_normal(mu, logvar)
        kl_weight = batch.get("kl_weight", self.config.kl_weight)
        if isinstance(kl_weight, Tensor):
            kl_weight = kl_weight.item()
        loss = recon + kl_weight * kl

        return {
            "loss": loss,
            "recon_loss": recon,
            "kl_loss": kl,
            "kl_weight": kl_weight,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "x_hat": x_hat,
            "gt_data": x,
            "padding_mask": padding_mask,
            "base_encoder_output": encoded,
            "base_decoder_output": {"x": x_hat},
            "batch": batch,
        }

    def forward(self, **batch) -> dict[str, Any]:
        return self.compute_loss(**batch)


if __name__ == "__main__":
    from torch import nn
    from torch.nn.functional import mse_loss

    batch_size = 8
    data_dim = 16
    latent_dim = 4

    class _EncoderWrapper(Module):
        def __init__(self, data_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim * 2),
            )
            self.latent_dim = latent_dim

        def forward(self, **batch: Any) -> dict[str, Tensor]:
            out = self.net(batch["gt_data"])
            return {
                "mu": out[..., : self.latent_dim],
                "logvar": out[..., self.latent_dim :],
            }

    class _DecoderWrapper(Module):
        def __init__(self, latent_dim: int, data_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, data_dim),
            )

        def forward(self, **batch: Any) -> dict[str, Tensor]:
            return {"x": self.net(batch["z"])}

    config = GaussianVAEConfig(ndim_micro_shape=1, latent_dim=latent_dim)
    vae = GaussianVAE(
        config=config,
        encoder=_EncoderWrapper(data_dim, latent_dim),
        decoder=_DecoderWrapper(latent_dim, data_dim),
        recon_loss_fn=lambda pred, target, mask: mse_loss(pred, target),
    )

    x = torch.randn(batch_size, data_dim)
    batch = {"gt_data": x, "padding_mask": torch.ones_like(x)}
    result = vae.compute_loss(**batch)
    result["loss"].backward()

    samples = vae.sampling((4, data_dim), device=x.device)
    print(f"loss={result['loss'].item():.4f}, recon={result['recon_loss'].item():.4f}, kl={result['kl_loss'].item():.4f}")
    print(f"sample shape={samples['x'].shape}")
