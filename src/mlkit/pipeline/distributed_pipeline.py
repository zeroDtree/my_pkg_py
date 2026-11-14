import logging
import os
import shutil
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import accelerate
import datasets
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ..util.decorators import inherit_docstrings
from ..util.report import defer_error, has_deferred_error, report
from .callback import BaseCallback, CallbackEvent
from .pipeline import BasePipeline, LogConfig, TrainingConfig


@inherit_docstrings
class DistributedTrainingConfig(TrainingConfig):
    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 4,
        device: str = "cuda",
        save_strategy: Literal["epochs", "steps", None] = "epochs",
        save_dir: str | None = None,
        save_steps: int = 10,
        save_epochs: int = 1,
        save_total_limit: int = 5,
        num_workers: int = 4,
        train_shuffle: bool = True,
        eval_strategy: Literal["epochs", "steps"] | None = None,
        eval_steps: int = 500,
        eval_epochs: int = 1,
        grad_clip_strategy: Literal["norm", "value", None] = "norm",
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        find_unused_parameters: bool = False,
        **kwargs,
    ):
        """Initialize the DistributedTrainingConfig

        Args:
            n_epochs (int, optional): the number of epochs. Defaults to 100.
            batch_size (int, optional): the batch size. Defaults to 4.
            device (str, optional): the device to use for training. Defaults to "cuda".
            save_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;, None], optional): the strategy determines whether to save the model and when to save it. Defaults to "epochs".
            save_dir (str, optional): the directory to save the model. Defaults to None.
            save_steps (int, optional): the number of steps to save the model. Defaults to 10.
            save_epochs (int, optional): the number of epochs to save the model. Defaults to 1.
            save_total_limit (int, optional): the maximum number of checkpoints to save. Defaults to 5.
            num_workers (int, optional): the number of workers to use for data loading. Defaults to 4.
            train_shuffle (bool, optional): whether to shuffle the training data. Defaults to True.
            eval_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;], optional): the strategy determines whether to evaluate the model and when to evaluate it. Defaults to None.
            eval_steps (int, optional): the number of steps to evaluate the model. Defaults to 500.
            eval_epochs (int, optional): the number of epochs to evaluate the model. Defaults to 1.
            grad_clip_strategy (Literal[&quot;norm&quot;, &quot;value&quot;, None], optional): the strategy determines whether to clip the gradient and how to clip it. Defaults to "norm".
            max_grad_norm (float, optional): the maximum gradient norm to clip the gradient. Defaults to 1.0.
            max_grad_value (float, optional): the maximum gradient value to clip the gradient. Defaults to 1.0.
            gradient_accumulation_steps (int, optional): the number of steps to accumulate gradients before updating the model. Defaults to 1.
            mixed_precision (str, optional): the mixed precision to use for training. Defaults to "fp16".
        """

        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            save_strategy=save_strategy,
            save_dir=save_dir,
            save_steps=save_steps,
            save_epochs=save_epochs,
            save_total_limit=save_total_limit,
            num_workers=num_workers,
            train_shuffle=train_shuffle,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            eval_epochs=eval_epochs,
            grad_clip_strategy=grad_clip_strategy,
            max_grad_norm=max_grad_norm,
            max_grad_value=max_grad_value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.mixed_precision = mixed_precision
        self.find_unused_parameters = find_unused_parameters


@inherit_docstrings
class DistributedPipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler],
        training_config: DistributedTrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger | None,
        collate_fn: Optional[Callable] = None,
        seed: int = 42,
        callbacks: Optional[List[BaseCallback]] = None,
        **kwargs,
    ):
        """Initialize the DistributedPipeline

        Args:
            model (torch.nn.Module): the model to train
            dataset (Union[torch.utils.data.Dataset, datasets.Dataset]): the dataset to train on
            optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]): the optimizers to use for training
            training_config (DistributedTrainingConfig): the training configuration
            log_config (LogConfig): the logging configuration
            logger (logging.Logger): the logger to use for logging
            collate_fn (Optional[Callable], optional): the collate function to use for the dataset. Defaults to None.
            seed (int, optional): the seed to use for the random number generator. Defaults to 42.
        """
        accelerate.utils.set_seed(seed)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            mixed_precision=training_config.mixed_precision,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=training_config.find_unused_parameters),
            ],
        )

        super().__init__(
            model=model,
            dataset=dataset,
            optimizers=optimizers,
            training_config=training_config,
            log_config=log_config,
            collate_fn=collate_fn,
            logger=logger,
            callbacks=callbacks,
            load_checkpoint=False,
            **kwargs,
        )

        # Prepare everything for distributed training BEFORE loading
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.scheduler
        )

        # Load checkpoint AFTER prepare to ensure proper state restoration
        if self.training_config.save_dir is not None and self.training_config.save_dir != "":
            self.load()
            self.training_config = training_config

        if self.accelerator.is_main_process:
            assert self.logger is not None, f"Error from {self.__class__.__name__}: logger is required"
            self.logger.info("Using distributed training with accelerate")
            self.logger.info(f"Number of processes: {self.accelerator.num_processes}")
            self.logger.info(f"Current device: {self.accelerator.device}")

    def gradient_clip(self) -> None:
        model = self.model
        if self.training_config.grad_clip_strategy == "norm":
            self.accelerator.clip_grad_norm_(
                model.parameters(),
                max_norm=self.training_config.max_grad_norm,
                norm_type=2,
            )
        if self.training_config.grad_clip_strategy == "value":
            self.accelerator.clip_grad_value_(
                model.parameters(),
                clip_value=self.training_config.max_grad_value,
            )

    def train_a_step(self, batch: Dict[str, Any]):
        self.trigger_callbacks(event=CallbackEvent.STEP_START, batch=batch)
        model: torch.nn.Module = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        logger = self.logger
        model.train()

        result = {}

        should_sync = (
            self.training_state.current_global_step + 1
        ) % self.training_config.gradient_accumulation_steps == 0
        ctx = self.accelerator.no_sync(model=model) if not should_sync else nullcontext()
        with ctx:
            raw_loss = self.compute_loss(model, batch)
            if isinstance(raw_loss, torch.Tensor):
                loss = raw_loss
            else:
                loss = cast(torch.Tensor, cast(Dict[str, Any], raw_loss)["loss"])
            self.trigger_callbacks(event=CallbackEvent.PRE_BACKWARD)
            self.accelerator.backward(loss)
            self.trigger_callbacks(event=CallbackEvent.POST_BACKWARD)
        if should_sync:
            self.gradient_clip()
            self.trigger_callbacks(event=CallbackEvent.PRE_OPTIMIZER_STEP)
            optimizer.step()
            self.trigger_callbacks(event=CallbackEvent.POST_OPTIMIZER_STEP)
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        result["loss"] = loss.item()
        result["lr"] = scheduler.get_last_lr()[0]
        result["global_step"] = self.training_state.current_global_step

        # Only the global main process has a reporter. A backend failure is
        # remembered, and every rank hears about it on a logging step.
        if self._log_interval_due("steps") and self.accelerator.is_main_process and logger is not None:
            report(result, step=self.training_state.current_global_step, logger=logger, on_error="defer")
        if self._log_interval_due("steps"):
            self._sync_halt()

        self.trigger_callbacks(event=CallbackEvent.STEP_END, batch=batch)
        return result

    def _share_flag(self, local: bool) -> bool:
        """Share one rank-local boolean with every rank.

        ``num_processes == 1`` returns ``local``. Otherwise the rank where
        ``local`` is true sets Accelerate's trigger, and every rank checks it.
        The check consumes the trigger. Every rank must enter this call
        together, and the flag must be consumed before the next phase sets it.

        Args:
            local: this rank's value.

        Returns:
            True when any rank passed true.
        """
        if self.accelerator.num_processes == 1:
            return local
        if local:
            self.accelerator.set_trigger()
        return bool(self.accelerator.check_trigger())

    def _halt_if_any_rank_failed(self) -> bool:
        """Share a deferred error stored on the main process.

        ``num_processes == 1`` reads the local error. Otherwise only the main
        process sets the trigger, and every rank checks it.
        """
        if self.accelerator.num_processes == 1:
            return has_deferred_error()
        return self._share_flag(self.accelerator.is_main_process and has_deferred_error())

    def _abort_uncommitted_save(self, temp_dir: str, error: Exception | None) -> None:
        """Drop an uncommitted checkpoint and stop every rank.

        Only the rank that caught ``error`` remembers it. The published
        checkpoint directory is left untouched.

        Args:
            temp_dir: temporary directory for this attempt.
            error: exception caught on this rank, or ``None`` when another rank
                failed.
        """
        if self.accelerator.is_main_process:
            if self.logger is not None:
                if error is not None:
                    self.logger.error(f"Failed to save checkpoint: {error}")
                else:
                    self.logger.error("Failed to save checkpoint on another rank")
            shutil.rmtree(temp_dir, ignore_errors=True)
        if error is not None:
            defer_error(error)
        self._halt = True

    def save(self) -> None:
        self.trigger_callbacks(event=CallbackEvent.PRE_SAVE)
        save_dir = self.training_config.save_dir
        if save_dir is None or save_dir == "":
            return

        epoch = self.training_state.current_epoch
        step = self.training_state.current_step_in_epoch
        global_step = self.training_state.current_global_step
        checkpoint_name = self._get_checkpoint_name(epoch, step, global_step)
        temp_checkpoint_dir = os.path.join(save_dir, f"tmp_{checkpoint_name}")
        final_checkpoint_dir = os.path.join(save_dir, checkpoint_name)

        # The global main decides whether this checkpoint is already published.
        already_saved = self.accelerator.is_main_process and os.path.exists(final_checkpoint_dir)
        if self._share_flag(already_saved):
            return

        prepare_error: Exception | None = None
        if self.accelerator.is_main_process:
            try:
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(temp_checkpoint_dir, exist_ok=True)
            except Exception as exc:
                prepare_error = exc
        if self._share_flag(prepare_error is not None):
            self._abort_uncommitted_save(temp_checkpoint_dir, prepare_error)
            return

        # save_state uses collectives. A failure inside that collective can
        # still leave the other ranks there; this joins ranks that have returned.
        state_error: Exception | None = None
        try:
            self.accelerator.save_state(temp_checkpoint_dir)
        except Exception as exc:
            state_error = exc
        if self._share_flag(state_error is not None):
            self._abort_uncommitted_save(temp_checkpoint_dir, state_error)
            return

        # Other ranks are done writing. Metadata and rename are one publish.
        publish_error: Exception | None = None
        if self.accelerator.is_main_process:
            try:
                for base_name in [
                    "training_state",
                    "training_config",
                    "log_config",
                ]:
                    file_path = os.path.join(temp_checkpoint_dir, f"{base_name}.pth")
                    torch.save(getattr(self, base_name), file_path)
                self._persist_report_run(temp_checkpoint_dir)
                os.rename(temp_checkpoint_dir, final_checkpoint_dir)
                if self.logger is not None:
                    self.logger.info(f"Model saved to {final_checkpoint_dir}")
            except Exception as exc:
                publish_error = exc
        if self._share_flag(publish_error is not None):
            self._abort_uncommitted_save(temp_checkpoint_dir, publish_error)
            return

        if self.accelerator.is_main_process:
            try:
                self._cleanup_old_checkpoints(save_dir=save_dir)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.error(f"Failed to delete old checkpoints: {exc}")
        if self._halt_if_any_rank_failed():
            self._halt = True
        self.trigger_callbacks(event=CallbackEvent.POST_SAVE)

    def load(self) -> None:
        self.trigger_callbacks(event=CallbackEvent.PRE_LOAD)
        # check load condition ============================================================================
        checkpoint_dir = self.get_latest_checkpoint_dir()
        if checkpoint_dir is None or len(os.listdir(checkpoint_dir)) <= 0:
            return

        # load ============================================================================================
        self.accelerator.load_state(checkpoint_dir, load_kwargs={"weights_only": False})

        # Load training metadata
        for base_name in ["training_state", "training_config", "log_config"]:
            file_path = os.path.join(checkpoint_dir, f"{base_name}.pth")
            if not os.path.exists(file_path):
                if self.accelerator.is_main_process and self.logger is not None:
                    self.logger.info(f"File {file_path} does not exist")
                continue
            setattr(self, base_name, torch.load(file_path, weights_only=False))

        if self.accelerator.is_main_process and self.logger is not None:
            self.logger.info(f"Model loaded from {checkpoint_dir}")
        self.trigger_callbacks(event=CallbackEvent.POST_LOAD)

    def trigger_callbacks(self, event: CallbackEvent, **kwargs):
        """Trigger all callbacks for a given event

        Args:
            event (CallbackEvent): the event to trigger
            **kwargs: the keyword arguments to pass to the callback
        """
        super().trigger_callbacks(
            event=event,
            accelerator=self.accelerator,
            **kwargs,
        )
