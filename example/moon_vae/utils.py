import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, cast

import torch
from omegaconf import DictConfig
from shared.dataset import MoonsDataset
from shared.model import mse_loss
from torch import Tensor
from torch.nn import Module

from mlkit.util.utils_for_main import (
    get_learing_rate_scheduler,  # noqa: F401
    get_new_save_dir,  # noqa: F401
    get_optimizer,  # noqa: F401
    get_run_name,  # noqa: F401
    get_train_class,  # noqa: F401
    load_checkpoint,  # noqa: F401
)


class MoonsVAEEncoder(Module):
    """4-layer MLP encoder for 2-D moons; reads ``batch["gt_data"]``."""

    def __init__(self, dim: int = 2, latent_dim: int = 4, h: int = 64) -> None:
        super().__init__()
        from torch import nn

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, latent_dim * 2),
        )
        self._init_logvar_bias()

    def _init_logvar_bias(self) -> None:
        from torch import nn

        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            return
        with torch.no_grad():
            nn.init.zeros_(last.bias)
            last.bias[self.latent_dim :].fill_(-2.0)

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        out = self.net(batch["gt_data"])
        d = self.latent_dim
        return {"mu": out[..., :d], "logvar": out[..., d:]}


class MoonsVAEDecoder(Module):
    """4-layer MLP decoder for 2-D moons; reads ``batch["z"]``."""

    def __init__(self, dim: int = 2, latent_dim: int = 4, h: int = 64) -> None:
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        return {"x": self.net(batch["z"])}


def get_dataset(cfg: DictConfig):
    train_dataset = MoonsDataset(n_samples=cfg.dataset.n_samples)
    return train_dataset, train_dataset, train_dataset


def get_collate_fn(cfg: DictConfig):
    import torch

    def collate_fn(examples):
        gt_data = torch.stack(examples)
        return {
            "gt_data": gt_data,
            "padding_mask": torch.ones_like(gt_data),
        }

    return collate_fn


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from hooks import build_kl_anneal_hook, build_vae_metrics_hook

    from mlkit.generative_model.vae import GaussianVAE, GaussianVAEConfig
    from mlkit.model.model_for_pipeline import ModelForPipeline

    latent_dim = cfg.vae.latent_dim
    vae_config = GaussianVAEConfig(
        ndim_micro_shape=cfg.vae.ndim_micro_shape,
        latent_dim=latent_dim,
        decoder_sigma=cfg.vae.decoder_sigma,
        kl_weight=cfg.vae.kl_weight,
    )
    vae = GaussianVAE(
        config=vae_config,
        encoder=MoonsVAEEncoder(latent_dim=latent_dim),
        decoder=MoonsVAEDecoder(latent_dim=latent_dim),
        recon_loss_fn=mse_loss,
    )
    pipeline_model = ModelForPipeline(model=vae)
    pipeline_model.register_hooks(
        [
            build_kl_anneal_hook(target=cfg.vae.kl_weight, anneal_steps=cfg.vae.kl_anneal_steps),
            build_vae_metrics_hook(log_interval=cfg.log.log_steps),
        ]
    )

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {"model": pipeline_model}
