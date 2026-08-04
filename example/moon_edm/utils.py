import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast

import torch
from omegaconf import DictConfig
from shared.dataset import MoonsDataset
from shared.model import (
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


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from mlkit.diffusion.conditioner.conditioner import LGDConditioner
    from mlkit.diffusion.euclidean_edm_diffuser import (
        EuclideanEDMConfig,
        EuclideanEDMDiffuser,
    )
    from mlkit.diffusion.time_scheduler import DiffusionTimeScheduler
    from mlkit.model.model_for_pipeline import ModelForPipeline
    from mlkit.util.mask.masker import Masker

    base_model = MoonsMLP()
    wrapped = MoonsModelWrapper(base_model=base_model, unsqueeze_t=False)

    time_scheduler = DiffusionTimeScheduler(
        num_train_timesteps=cfg.gm.n_discretization_steps,
    )
    gm_config = EuclideanEDMConfig(
        n_discretization_steps=cfg.gm.n_discretization_steps,
        ndim_micro_shape=1,
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
        model=wrapped,
        masker=Masker(ndim_mini_micro_shape=0),
        loss_fn=mse_loss,
    )

    classifier = train_moons_classifier()
    classifier = classifier.to(get_device_from_accelerator())

    classifier_conditioner = build_classifier_conditioner(
        base_conditioner_class=LGDConditioner,
        classifier_model=classifier,
        guidance_scale=cfg.gm.gs,
    )
    sampling_hook = gm.get_condition_pre_update_in_step_fn_hook([classifier_conditioner])
    sampling_hook_handlers = gm.register_hooks([sampling_hook])
    train_hook = gm.get_condition_post_compute_loss_hook([classifier_conditioner])
    train_hook_handlers = gm.register_hooks([train_hook])

    pipeline_model = ModelForPipeline(model=gm)

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        pipeline_model = cast(ModelForPipeline, load_checkpoint(pipeline_model, final_model_ckpt_path))

    return {
        "model": pipeline_model,
        "train_hook_handlers": train_hook_handlers,
        "sampling_hook_handlers": sampling_hook_handlers,
    }
