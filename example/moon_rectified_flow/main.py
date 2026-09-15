import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.runner import run_experiment
from utils import get_collate_fn, get_dataset, get_model


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    from typing import Any, cast

    import matplotlib.pyplot as plt
    import torch
    from omegaconf import OmegaConf
    from shared.dataset import MoonsDataset
    from torch import Tensor
    from utils import get_conditional_collate_fn, get_conditional_model

    from mlkit.pipeline.pipeline import LogConfig
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.utils_for_main import (
        get_learing_rate_scheduler,
        get_new_save_dir,
        get_optimizer,
        get_train_class,
    )

    model = model_result["model"].model
    sampling_hook_handlers = model_result["sampling_hook_handlers"]
    model = model.to(accelerator.device)
    n_samples = 256
    n_steps = cfg.flow.n_discretization_steps
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    # --- Unconditional sampling ---
    for handler in sampling_hook_handlers:
        handler.disable()
    result: dict = model.sampling(shape=(n_samples, 2), device=accelerator.device, return_all=True)
    x_list = [x.detach().cpu() for x in result["x_list"]]

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    axes[0].scatter(x_list[0][:, 0], x_list[0][:, 1], s=10)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    for i in range(n_steps):
        x = x_list[i + 1]
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")
    fig.tight_layout()
    fig.savefig("sampling_unconditional.png")
    plt.close(fig)

    # --- Classifier-guided conditional sampling ---
    for handler in sampling_hook_handlers:
        handler.enable()
    sigma = 1.0
    x_noise = torch.randn(n_samples, 2) * sigma
    c_eval = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)
    result_c: dict = model.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
        sampling_condition=c_eval,
    )
    x_list_c = [x.detach().cpu() for x in result_c["x_list"]]
    colors = ["blue" if lbl == 0 else "orange" for lbl in c_eval.squeeze().tolist()]

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4), sharex=True, sharey=True)
    axes[0].scatter(x_noise[:, 0], x_noise[:, 1], s=10, c=colors)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    for i in range(n_steps):
        x = x_list_c[i + 1]
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10, c=colors)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")
        axes[i + 1].set_xlim(-3.0, 3.0)
        axes[i + 1].set_ylim(-3.0, 3.0)
    fig.tight_layout()
    fig.savefig("sampling_condition.png")
    plt.close(fig)

    # --- Example-style conditional flow: train from scratch then sample ---
    cond_result = get_conditional_model(cfg)
    cond_pipeline_model = cond_result["model"]

    cond_dataset = MoonsDataset(n_samples=256)
    cond_collate = get_conditional_collate_fn(cfg)

    cond_opt = get_optimizer(cond_pipeline_model, cfg)
    cond_sched = get_learing_rate_scheduler(cond_opt, accelerator, cond_dataset, cfg)

    PipelineClass, TrainingConfigClass = get_train_class()
    train_kwargs: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(cfg.train, resolve=True))
    train_kwargs["save_dir"] = get_new_save_dir("checkpoints", cfg, suffix="_cfm")
    cond_training_config = TrainingConfigClass(**train_kwargs)
    log_kwargs: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(cfg.log, resolve=True))
    log_config = LogConfig(**log_kwargs)

    cond_pipeline = PipelineClass(
        model=cond_pipeline_model,
        train_dataset=cond_dataset,
        eval_dataset=cond_dataset,
        optimizers=(cond_opt, cond_sched),
        training_config=cond_training_config,
        log_config=log_config,
        logger=get_logger(name="cond_experiment", log_dir=get_and_create_new_log_dir(cfg.log.log_dir)),
        collate_fn=cond_collate,
    )
    cond_pipeline.train()

    # Sample with the framework
    cond_flow = cond_pipeline_model.model  # RectifiedFlow (trained)
    c_cfm = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)
    colors_cfm = ["blue" if lbl == 0 else "orange" for lbl in c_cfm.squeeze().tolist()]

    sampling_result = cond_flow.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
        sampling_condition=c_cfm,
    )
    x_list: list[Tensor] = sampling_result["x_list"]

    cfm_plot_every = 20
    cfm_plot_indices = list(range(0, len(x_list), cfm_plot_every))
    if cfm_plot_indices[-1] != len(x_list) - 1:
        cfm_plot_indices.append(len(x_list) - 1)
    cfm_time_steps = torch.linspace(0, 1.0, len(x_list))

    fig, axes = plt.subplots(
        1,
        len(cfm_plot_indices),
        figsize=(4 * len(cfm_plot_indices), 4),
        sharex=True,
        sharey=True,
    )
    for plot_count, idx in enumerate(cfm_plot_indices):
        x_plot = x_list[idx].detach().cpu()
        axes[plot_count].scatter(x_plot[:, 0], x_plot[:, 1], s=10, c=colors_cfm)
        axes[plot_count].set_title(f"t = {cfm_time_steps[idx]:.2f}")
        axes[plot_count].set_xlim(-3.0, 3.0)
        axes[plot_count].set_ylim(-3.0, 3.0)

    fig.tight_layout()
    fig.savefig("conditional_flow_matching.png")
    plt.close(fig)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(
        cfg,
        get_model,
        get_dataset,
        get_collate_fn,
        post_train_fn=post_train,
        disable_train_hooks=True,
    )


if __name__ == "__main__":
    main()
