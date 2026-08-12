import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, cast

from omegaconf import DictConfig
from shared.dataset import MoonsDataset
from shared.model import build_gan_metrics_hook
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.parametrizations import spectral_norm

from mlkit.util.utils_for_main import (
    get_learing_rate_scheduler,  # noqa: F401
    get_new_save_dir,  # noqa: F401
    get_optimizer,  # noqa: F401
    get_run_name,  # noqa: F401
    get_train_class,  # noqa: F401
    load_checkpoint,  # noqa: F401
)


class MoonsGenerator(Module):
    """4-layer MLP generator for 2-D moons; reads ``batch["z"]``."""

    def __init__(self, latent_dim: int = 2, dim: int = 2, h: int = 64) -> None:
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


class MoonsDiscriminator(Module):
    """4-layer MLP discriminator with spectral normalization; reads ``batch["gt_data"]``."""

    def __init__(self, dim: int = 2, h: int = 64) -> None:
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim, h)),
            nn.ELU(),
            spectral_norm(nn.Linear(h, h)),
            nn.ELU(),
            spectral_norm(nn.Linear(h, h)),
            nn.ELU(),
            spectral_norm(nn.Linear(h, 1)),
        )

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        return {"logits": self.net(batch["gt_data"]).squeeze(-1)}


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
    from mlkit.generative_model.gan import GAN, GANConfig
    from mlkit.model.model_for_pipeline import ModelForPipeline

    latent_dim = cfg.gan.latent_dim
    gan_config = GANConfig(
        ndim_micro_shape=cfg.gan.ndim_micro_shape,
        latent_dim=latent_dim,
        n_critic=cfg.gan.n_critic,
        r1_gamma=cfg.gan.r1_gamma,
        ema_decay=cfg.gan.ema_decay,
    )
    gan = GAN(
        config=gan_config,
        generator=MoonsGenerator(latent_dim=latent_dim, h=cfg.gan.hidden_dim),
        discriminator=MoonsDiscriminator(h=cfg.gan.hidden_dim),
    )
    pipeline_model = ModelForPipeline(model=gan)
    pipeline_model.register_hooks([build_gan_metrics_hook(log_interval=cfg.log.log_steps)])

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {"model": pipeline_model}
