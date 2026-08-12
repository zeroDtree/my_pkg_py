import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

import torch
from omegaconf import DictConfig
from shared.dataset import get_hf_image_dataset
from shared.model import build_unet2d, mse_loss
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


def get_dataset(cfg: DictConfig):
    return get_hf_image_dataset(cfg)


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from mlkit.diffusion.euclidean_edm_diffuser import (
        EuclideanEDMConfig,
        EuclideanEDMDiffuser,
    )
    from mlkit.diffusion.time_scheduler import DiffusionTimeScheduler
    from mlkit.model.model_for_pipeline import ModelForPipeline
    from mlkit.util.mask.image_masker import ImageMasker

    if model is None:
        model = build_unet2d(cfg.dataset.image_size)

    class BaseModel(Module):
        def __init__(self, unet: Any):
            Module.__init__(self)
            self.model = unet

        def forward(
            self,
            x_t: Tensor,
            t: Tensor,
            padding_mask: Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> dict:
            p_noise: Tensor = self.model(x_t, t.squeeze(), return_dict=False)[0]
            return {"x": p_noise}

    model = BaseModel(unet=model)

    time_scheduler = DiffusionTimeScheduler(
        num_train_timesteps=cfg.gm.n_discretization_steps,
    )

    gm_config = EuclideanEDMConfig(
        n_discretization_steps=cfg.gm.n_discretization_steps,
        ndim_micro_shape=3,
        P_mean=cfg.gm.P_mean,
        P_std=cfg.gm.P_std,
        sigma_data=cfg.gm.sigma_data,
        sigma_min=cfg.gm.sigma_min,
        sigma_max=cfg.gm.sigma_max,
        rho=cfg.gm.rho,
    )
    gm = EuclideanEDMDiffuser(
        config=gm_config,
        time_scheduler=time_scheduler,
        model=model,
        masker=ImageMasker(),
        loss_fn=mse_loss,
    )

    pipeline_model = ModelForPipeline(model=gm)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = load_checkpoint(pipeline_model, final_model_ckpt_path)

    return {"model": pipeline_model}


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        batch = [ex["images"] for ex in examples]
        gt_data = torch.stack(batch)
        return {
            "gt_data": gt_data,
            "padding_mask": torch.ones_like(gt_data),
        }

    return collate_fn
