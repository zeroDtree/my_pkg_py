"""Shared training loop for all examples.

Every example's ``main()`` delegates to ``run_experiment()``, passing only what
is unique: how to build the model, dataset, collate function, and optional
post-training visualisation.
"""

from collections.abc import Callable
from typing import Any, cast

from omegaconf import DictConfig


def run_experiment(
    cfg: DictConfig,
    get_model_fn: Callable,
    get_dataset_fn: Callable,
    get_collate_fn: Callable,
    post_train_fn: Callable | None = None,
    save_dir_suffix: str = "",
    disable_train_hooks: bool = True,
    get_optimizer_fn: Callable | None = None,
) -> None:
    """Run a complete train → (optional) visualise cycle.

    Args:
        cfg: Hydra config.
        get_model_fn: ``(cfg, **kwargs) -> dict | Module``.
            Must return either a ``dict`` with at least ``{"model": ...}`` or a
            bare ``Module``.  Dicts may also carry ``train_hook_handlers`` and
            ``sampling_hook_handlers``.
        get_dataset_fn: ``(cfg) -> (train, val, test)``.
        get_collate_fn: ``(cfg) -> collate_fn``.
        post_train_fn: Optional ``(cfg, model_result, pipeline, accelerator,
            train_set) -> None`` called on the main process after training.
        save_dir_suffix: Appended to the checkpoint save-directory name.
        disable_train_hooks: If True (default), disable ``train_hook_handlers``
            before training so the model trains unconditionally.  Set False for
            conditional-training examples (e.g. ``conditional_fm``).
        get_optimizer_fn: Optional ``(model, cfg) -> Optimizer``. Defaults to
            ``get_optimizer`` from ``mlkit.util.utils_for_main``.
    """
    import wandb
    from accelerate import Accelerator
    from omegaconf import OmegaConf

    from mlkit.pipeline.pipeline import LogConfig
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.seed import seed_everything
    from mlkit.util.show import show_info
    from mlkit.util.utils_for_main import (
        get_learing_rate_scheduler,
        get_new_save_dir,
        get_optimizer,
        get_run_name,
        get_train_class,
    )

    seed_everything(cfg.train.seed)

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    print(f"accelerator.device = {accelerator.device}")

    logger = None
    if accelerator.is_local_main_process:
        log_dir = get_and_create_new_log_dir(cfg.log.log_dir)
        logger = get_logger(name="experiment", log_dir=log_dir)
        logger.info(f"accelerator.device = {accelerator.device}")
        logger.info(f"seed = {cfg.train.seed}")
        run_name = get_run_name(cfg)
        logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
        wandb.init(
            reinit=cfg.wandb.reinit,
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            name=run_name,
            group=cfg.wandb.group,
            entity=cfg.wandb.entity,
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        )

    # Build model; normalise to dict so downstream code is uniform.
    model_result = get_model_fn(cfg)
    if not isinstance(model_result, dict):
        model_result = {"model": model_result}
    model = model_result["model"]

    if disable_train_hooks:
        for handler in model_result.get("train_hook_handlers", []):
            handler.disable()

    train_set, val_set, _ = get_dataset_fn(cfg)
    optimizer = (get_optimizer_fn or get_optimizer)(model, cfg)
    lr_scheduler = get_learing_rate_scheduler(optimizer, accelerator, train_set, cfg)
    show_info(model=model, optimizer=optimizer)

    log_config = LogConfig(**cfg.log)
    PipelineClass, TrainingConfigClass = get_train_class()
    training_config = TrainingConfigClass(**cfg.train)
    if accelerator.is_local_main_process:
        training_config.save_dir = get_new_save_dir(training_config.save_dir, cfg, suffix=save_dir_suffix)

    print(training_config.__dict__)

    pipeline = PipelineClass(
        model=model,
        train_dataset=train_set,
        eval_dataset=val_set,
        optimizers=(optimizer, lr_scheduler),
        training_config=training_config,
        log_config=log_config,
        collate_fn=get_collate_fn(cfg),
        logger=logger,
    )

    pipeline.train()

    if accelerator.is_local_main_process:
        wandb.finish()

    if accelerator.is_local_main_process and post_train_fn is not None:
        post_train_fn(cfg, model_result, pipeline, accelerator, train_set)
