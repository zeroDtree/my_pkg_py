import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import DictConfig
from shared.dataset import get_hf_image_dataset
from shared.model import ImageModelWrapper, build_unet2d, mse_loss

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
    from mlkit.diffusion.euclidean_vpsde_diffuser import (
        EuclideanVPSDEConfig,
        EuclideanVPSDEDiffuser,
    )
    from mlkit.diffusion.time_scheduler import DiffusionTimeScheduler
    from mlkit.util.mask.image_masker import ImageMasker

    if model is None:
        model = build_unet2d(cfg.dataset.image_size)

    wrapped = ImageModelWrapper(unet=model)
    time_scheduler = DiffusionTimeScheduler(
        continuous_time_start=0.0,
        continuous_time_end=1.0,
        num_train_timesteps=cfg.diffusion.n_discretization_steps,
        num_inference_steps=cfg.diffusion.get("n_inference_steps", None),
    )

    diffusion_config = EuclideanVPSDEConfig(
        n_discretization_steps=cfg.diffusion.n_discretization_steps,
        ndim_micro_shape=3,
        n_inference_steps=cfg.diffusion.get("n_inference_steps", None),
        n_correct_steps=cfg.diffusion.get("n_correct_steps", 0),
        snr=cfg.diffusion.get("snr", 1.0),
    )
    diffuser = EuclideanVPSDEDiffuser(
        config=diffusion_config,
        time_scheduler=time_scheduler,
        loss_fn=mse_loss,
        masker=ImageMasker(),
        model=wrapped,
    )

    from mlkit.model.model_for_pipeline import ModelForPipeline

    pipeline_model = ModelForPipeline(model=diffuser)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = load_checkpoint(pipeline_model, final_model_ckpt_path)

    return {"model": pipeline_model}


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        batch = [ex["images"] for ex in examples]
        gt_data = torch.stack(batch)
        return {
            "gt_data": gt_data,
            "padding_mask": torch.ones_like(gt_data, dtype=torch.bool),
        }

    return collate_fn
