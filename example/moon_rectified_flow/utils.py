import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast

import torch
from omegaconf import DictConfig
from shared.dataset import MoonsDataset
from shared.model import (
    ConditionalMoonsMLP,
    ConditionalMoonsModelWrapper,
    MoonsMLP,
    MoonsModelWrapper,
    build_classifier_conditioner,
    get_device_from_accelerator,
    mse_loss,
    train_moons_classifier,
)

from mlkit.util.utils_for_main import (
    get_learing_rate_scheduler,  # noqa: F401
    get_new_save_dir,  # noqa: F401
    get_optimizer,  # noqa: F401
    get_run_name,  # noqa: F401
    get_train_class,  # noqa: F401
    load_checkpoint,  # noqa: F401
)


def get_dataset(cfg: DictConfig):
    train_dataset = MoonsDataset(n_samples=256)
    return train_dataset, train_dataset, train_dataset


def get_collate_fn(cfg: DictConfig):
    from sklearn.datasets import make_moons
    from torch import Tensor

    def collate_fn(examples):
        x_1 = torch.stack(examples)
        return {
            "gt_data": Tensor(make_moons(256, noise=0.15)[0]),
            "padding_mask": torch.ones_like(x_1),
        }

    return collate_fn


def get_conditional_collate_fn(cfg: DictConfig):
    from sklearn.datasets import make_moons
    from torch import Tensor

    def collate_fn(examples):
        x_1 = torch.stack(examples)
        x_data, c_data = make_moons(256, noise=0.15)
        return {
            "gt_data": Tensor(x_data),
            "c": Tensor(c_data).view(-1, 1),
            "padding_mask": torch.ones_like(x_1),
        }

    return collate_fn


def get_conditional_model(cfg: DictConfig, final_model_ckpt_path=None):
    from mlkit.flow_matching.rectified_flow import RectifiedFlow, RectifiedFlowConfig
    from mlkit.flow_matching.time_scheduler import FlowMatchingTimeScheduler
    from mlkit.model.model_for_pipeline import ModelForPipeline
    from mlkit.util.mask.image_masker import ImageMasker

    base_model = ConditionalMoonsMLP()
    wrapped = ConditionalMoonsModelWrapper(base_model=base_model)

    time_scheduler = FlowMatchingTimeScheduler(
        num_train_timesteps=cfg.flow.n_discretization_steps,
        num_inference_steps=100,
    )
    flow_config = RectifiedFlowConfig(
        n_discretization_steps=cfg.flow.n_discretization_steps,
        ndim_micro_shape=1,
        n_inference_steps=100,
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


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from mlkit.flow_matching.conditioner import LGFMConditioner
    from mlkit.flow_matching.rectified_flow import RectifiedFlow, RectifiedFlowConfig
    from mlkit.flow_matching.time_scheduler import FlowMatchingTimeScheduler
    from mlkit.model.model_for_pipeline import ModelForPipeline
    from mlkit.util.mask.image_masker import ImageMasker

    base_model = MoonsMLP()
    wrapped = MoonsModelWrapper(base_model=base_model, unsqueeze_t=True)

    time_scheduler = FlowMatchingTimeScheduler(
        num_train_timesteps=cfg.flow.n_discretization_steps,
        num_inference_steps=cfg.flow.n_inference_steps,
    )
    flow_config = RectifiedFlowConfig(
        n_discretization_steps=cfg.flow.n_discretization_steps,
        ndim_micro_shape=1,
        n_inference_steps=cfg.flow.n_inference_steps,
    )
    flow = RectifiedFlow(
        config=flow_config,
        time_scheduler=time_scheduler,
        model=wrapped,
        masker=ImageMasker(ndim_mini_micro_shape=0),
        loss_fn=mse_loss,
    )

    classifier = train_moons_classifier()
    classifier = classifier.to(get_device_from_accelerator())

    classifier_conditioner = build_classifier_conditioner(
        base_conditioner_class=LGFMConditioner,
        classifier_model=classifier,
        guidance_scale=10.0,
    )
    sampling_hook = flow.get_condition_pre_update_in_step_fn_hook([classifier_conditioner])
    sampling_hook_handlers = flow.register_hooks([sampling_hook])
    train_hook = flow.get_condition_post_compute_loss_hook([classifier_conditioner])
    train_hook_handlers = flow.register_hooks([train_hook])

    pipeline_model = ModelForPipeline(model=flow)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {
        "model": pipeline_model,
        "train_hook_handlers": train_hook_handlers,
        "sampling_hook_handlers": sampling_hook_handlers,
    }
