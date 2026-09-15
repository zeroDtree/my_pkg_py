import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast

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
    from mlkit.flow_matching.rectified_flow import RectifiedFlow, RectifiedFlowConfig
    from mlkit.flow_matching.time_scheduler import FlowMatchingTimeScheduler
    from mlkit.model.model_for_pipeline import ModelForPipeline
    from mlkit.util.mask.image_masker import ImageMasker

    if model is None:
        model = build_unet2d(cfg.dataset.image_size)

    wrapped = ImageModelWrapper(unet=model)
    time_scheduler = FlowMatchingTimeScheduler(
        num_train_timesteps=cfg.flow.n_discretization_steps,
        num_inference_steps=cfg.flow.n_inference_steps,
    )

    flow_config = RectifiedFlowConfig(
        n_discretization_steps=cfg.flow.n_discretization_steps,
        ndim_micro_shape=3,
        n_inference_steps=cfg.flow.n_inference_steps,
    )
    flow = RectifiedFlow(
        config=flow_config,
        time_scheduler=time_scheduler,
        model=wrapped,
        masker=ImageMasker(ndim_mini_micro_shape=0),
        loss_fn=mse_loss,
    )

    pipeline_model = ModelForPipeline(model=flow)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {"model": pipeline_model}


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        batch = [ex["images"] for ex in examples]
        return {"x_1": torch.stack(batch)}

    return collate_fn
