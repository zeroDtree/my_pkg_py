import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, cast

from omegaconf import DictConfig
from shared.dataset import MoonsDataset
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


class MoonsEnergyNet(Module):
    """4-layer MLP energy network for 2-D moons; reads ``batch["gt_data"]``."""

    def __init__(self, dim: int = 2, h: int = 64) -> None:
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, 1),
        )

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        return {"energy": self.net(batch["gt_data"]).squeeze(-1)}


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
    from mlkit.generative_model.ebm import EnergyBasedModel, EnergyBasedModelConfig
    from mlkit.model.model_for_pipeline import ModelForPipeline

    ebm_config = EnergyBasedModelConfig(
        ndim_micro_shape=cfg.ebm.ndim_micro_shape,
        sigma=cfg.ebm.sigma,
        langevin_steps=cfg.ebm.langevin_steps,
        langevin_step_size=cfg.ebm.langevin_step_size,
    )
    ebm = EnergyBasedModel(
        config=ebm_config,
        energy_net=MoonsEnergyNet(h=cfg.ebm.hidden_dim),
    )
    pipeline_model = ModelForPipeline(model=ebm)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {"model": pipeline_model}
