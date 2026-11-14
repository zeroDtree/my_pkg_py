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
    after_train_fn: Callable | None = None,
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
            train_set) -> None`` called on the global main process after training.
            The process group is destroyed before this runs. The reporter stays
            open until this returns, so a failure here marks the run failed.
        after_train_fn: Optional ``(cfg, model_result, pipeline, accelerator,
            train_set) -> None`` called on every rank after training and before
            the process group is destroyed. A failure here destroys the group
            and skips ``post_train_fn``. When the pipeline has already halted
            because a rank-local failure was shared, every rank skips this
            callback. ``_finish`` then destroys the group and raises that
            failure on the rank that stored it.
        save_dir_suffix: Appended to the checkpoint save-directory name.
        disable_train_hooks: If True (default), disable ``train_hook_handlers``
            before training so the model trains unconditionally.  Set False for
            conditional-training examples (e.g. ``conditional_fm``).
        get_optimizer_fn: Optional ``(model, cfg) -> Optimizer``. Defaults to
            ``get_optimizer`` from ``mlkit.util.utils_for_main``.
    """
    from contextlib import ExitStack

    from accelerate import Accelerator
    from omegaconf import OmegaConf

    from mlkit.pipeline.pipeline import LogConfig, latest_checkpoint_dir
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.report import (
        clear_deferred_error,
        open_reporter,
        raise_deferred_error,
        resume_run_id,
        store_run_config,
        use_reporter,
    )
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
    run_name = ""

    def _train():
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
        if accelerator.is_main_process:
            effective_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            store_run_config(effective_config)
            if logger is not None:
                logger.info("Config after scheduler rewrite:\n" + OmegaConf.to_yaml(cfg))
        show_info(model=model, optimizer=optimizer)

        log_config = LogConfig(**cfg.log)
        PipelineClass, TrainingConfigClass = get_train_class()
        training_config = TrainingConfigClass(**cfg.train)
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
        return model_result, pipeline, train_set

    def _run_train() -> tuple[Any, Any, Any]:
        trained = _train()
        model_result, pipeline, train_set = trained
        # ``_halt`` is already the same on every rank: the training loop shared it.
        if after_train_fn is not None and not pipeline._halt:
            after_train_fn(cfg, model_result, pipeline, accelerator, train_set)
        return trained

    def _destroy_group() -> None:
        try:
            accelerator.end_training()
        except Exception:
            if logger is not None:
                logger.exception("Failed to destroy the process group")

    def _finish(trained: tuple[Any, Any, Any]) -> None:
        model_result, pipeline, train_set = trained
        # All ranks tear the process group down together, before main-process sampling.
        accelerator.wait_for_everyone()
        accelerator.end_training()
        raise_deferred_error()
        if accelerator.is_main_process and post_train_fn is not None:
            post_train_fn(cfg, model_result, pipeline, accelerator, train_set)

    def _reporter_open_failed(open_error: BaseException | None) -> bool:
        """Return whether any rank failed before training could start.

        Every rank calls this before ``prepare()``. The main process opens the
        reporter first, and the other ranks wait in this collective. A failure
        is shared so every rank can destroy the process group.
        """
        if accelerator.num_processes == 1:
            return open_error is not None
        if open_error is not None:
            accelerator.set_trigger()
        return bool(accelerator.check_trigger())

    clear_deferred_error()
    with ExitStack() as stack:
        open_error: BaseException | None = None
        if accelerator.is_main_process:
            try:
                log_dir = get_and_create_new_log_dir(cfg.log.log_dir)
                logger = get_logger(name="experiment", log_dir=log_dir)
                logger.info(f"accelerator.device = {accelerator.device}")
                logger.info(f"seed = {cfg.train.seed}")
                run_name = get_run_name(cfg)
                logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
                resolved_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
                reporter_kind = str(cfg.reporter).strip().lower()
                checkpoint_dir = latest_checkpoint_dir(
                    get_new_save_dir(cfg.train.save_dir, cfg, suffix=save_dir_suffix)
                )
                active_reporter = stack.enter_context(
                    open_reporter(
                        reporter_kind,
                        run_name=run_name,
                        config=resolved_config,
                        mlflow=cast(dict[str, Any], OmegaConf.to_container(cfg.mlflow, resolve=True)),
                        wandb=cast(dict[str, Any], OmegaConf.to_container(cfg.wandb, resolve=True)),
                        resume_run_id=resume_run_id(checkpoint_dir, reporter_kind),
                    )
                )
                stack.enter_context(use_reporter(active_reporter))
            except BaseException as exc:
                open_error = exc
        if _reporter_open_failed(open_error):
            _destroy_group()
            if open_error is not None:
                raise open_error
            raise RuntimeError("Reporter failed to open on another rank")
        try:
            trained = _run_train()
        except BaseException:
            _destroy_group()
            raise
        _finish(trained)
