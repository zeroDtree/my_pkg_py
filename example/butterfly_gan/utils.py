import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, cast

import torch
from omegaconf import DictConfig
from shared.dataset import get_hf_image_dataset
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


def _deconv_block(in_channels: int, out_channels: int) -> Module:
    from torch import nn

    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _spectral_conv_block(in_channels: int, out_channels: int) -> Module:
    from torch import nn

    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ButterflyGenerator(Module):
    """Convolutional generator for butterfly images; reads ``batch["z"]``."""

    def __init__(
        self,
        image_size: int = 128,
        out_channels: int = 3,
        latent_dim: int = 128,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        from torch import nn

        self.spatial_size = image_size // 16
        in_channels = base_channels * 8
        flat_dim = in_channels * self.spatial_size * self.spatial_size

        self.fc = nn.Linear(latent_dim, flat_dim)
        self.deconv = nn.Sequential(
            _deconv_block(in_channels, base_channels * 4),
            _deconv_block(base_channels * 4, base_channels * 2),
            _deconv_block(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        z = batch["z"]
        features = self.fc(z)
        features = features.view(
            z.shape[0],
            -1,
            self.spatial_size,
            self.spatial_size,
        )
        return {"x": self.deconv(features)}


class ButterflyDiscriminator(Module):
    """Convolutional discriminator with spectral normalization; reads ``batch["gt_data"]``."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        from torch import nn

        out_channels = base_channels * 8

        self.conv = nn.Sequential(
            _spectral_conv_block(in_channels, base_channels),
            _spectral_conv_block(base_channels, base_channels * 2),
            _spectral_conv_block(base_channels * 2, base_channels * 4),
            _spectral_conv_block(base_channels * 4, out_channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = spectral_norm(nn.Linear(out_channels, 1))

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        features = self.conv(batch["gt_data"])
        pooled = self.pool(features).flatten(start_dim=1)
        return {"logits": self.fc(pooled).squeeze(-1)}


def get_dataset(cfg: DictConfig):
    return get_hf_image_dataset(cfg)


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        batch = [ex["images"] for ex in examples]
        gt_data = torch.stack(batch)
        return {
            "gt_data": gt_data,
            "padding_mask": torch.ones_like(gt_data),
        }

    return collate_fn


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from mlkit.generative_model.gan import GAN, GANConfig
    from mlkit.model.model_for_pipeline import ModelForPipeline

    image_size = cfg.dataset.image_size
    latent_dim = cfg.gan.latent_dim
    base_channels = cfg.gan.base_channels
    gan_config = GANConfig(
        ndim_micro_shape=cfg.gan.ndim_micro_shape,
        latent_dim=latent_dim,
        n_critic=cfg.gan.n_critic,
        r1_gamma=cfg.gan.r1_gamma,
        ema_decay=cfg.gan.ema_decay,
    )
    gan = GAN(
        config=gan_config,
        generator=ButterflyGenerator(
            image_size=image_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
        ),
        discriminator=ButterflyDiscriminator(base_channels=base_channels),
    )
    pipeline_model = ModelForPipeline(model=gan)
    pipeline_model.register_hooks([build_gan_metrics_hook(log_interval=cfg.log.log_steps)])

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {"model": pipeline_model}
