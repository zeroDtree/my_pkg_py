import logging
import os
import shutil
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import accelerate
import datasets
import torch
import wandb
from accelerate import Accelerator

from ls_mlkit.my_pipeline.pipeline import BasePipeline, LogConfig, TrainingConfig
from ls_mlkit.my_utils.decorators import inherit_docstrings


@inherit_docstrings
class DistributedTrainingConfig(TrainingConfig):
    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 4,
        device: str = "cuda",
        save_strategy: Literal["epochs", "steps", None] = "epochs",
        save_dir: str = None,
        save_steps: int = 10,
        save_epochs: int = 1,
        save_total_limit: int = 5,
        num_workers: int = 4,
        train_shuffle: bool = True,
        eval_strategy: Literal["epochs", "steps"] = None,
        eval_steps: int = 500,
        eval_epochs: int = 1,
        grad_clip_strategy: Literal["norm", "value", None] = "norm",
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        *args,
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
            *args,
            **kwargs,
        )
        self.mixed_precision = mixed_precision


@inherit_docstrings
class DistributedPipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        training_config: DistributedTrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger,
        collate_fn: Optional[Callable] = None,
        seed: int = 42,
        *args,
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
        )

        super().__init__(
            model=model,
            dataset=dataset,
            optimizers=optimizers,
            training_config=training_config,
            log_config=log_config,
            collate_fn=collate_fn,
            logger=logger,
            *args,
            **kwargs,
        )

        # Prepare everything for distributed training
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        if self.accelerator.is_main_process:
            assert self.logger is not None, f"Error from {self.__class__.__name__}: logger is required"
            self.logger.info(f"Using distributed training with accelerate")
            self.logger.info(f"Number of processes: {self.accelerator.num_processes}")
            self.logger.info(f"Current device: {self.accelerator.device}")

    def gradient_clip(self) -> None:
        model = self.model
        if self.training_config.grad_clip_strategy == "norm":
            self.accelerator.clip_grad_norm_(
                model.parameters(), max_norm=self.training_config.max_grad_norm, norm_type=2
            )
        if self.training_config.grad_clip_strategy == "value":
            self.accelerator.clip_grad_value_(model.parameters(), clip_value=self.training_config.max_grad_value)

    def train_a_step(self, batch: Dict[str, Any]):
        model: torch.nn.Module = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        logger = self.logger
        model.train()

        result = {}

        if (self.training_state.current_global_step % self.training_config.gradient_accumulation_steps) < (
            self.training_config.gradient_accumulation_steps - 1
        ):
            with self.accelerator.no_sync(model=model):
                loss = self.compute_loss(model, batch)
                self.accelerator.backward(loss)
        else:
            loss = self.compute_loss(model, batch)
            self.accelerator.backward(loss)
            result["grad_norm_pre_clip"] = self.observer.get_gradient_norm()
            self.gradient_clip()
            result["grad_norm_post_clip"] = self.observer.get_gradient_norm()
            # print(f"grad_norm_pre_clip = {result['grad_norm_pre_clip']}, grad_norm_post_clip = {result['grad_norm_post_clip']}")

            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        result["loss"] = loss.item()
        result["weight_norm"] = self.observer.get_weight_norm()
        result["lr"] = scheduler.get_last_lr()[0]

        # Only log on main process
        if self._can_log(flag="steps") and self.accelerator.is_local_main_process:
            logger.info(
                f"[Training] Epoch {self.training_state.current_epoch}, Step {self.training_state.current_step_in_epoch}, Loss {loss.item()}"
            )
            wandb.log(result)

        return result

    def save(self) -> None:
        if not self.accelerator.is_main_process:
            return

        save_dir = self.training_config.save_dir
        if save_dir is None or save_dir == "":
            return
        os.makedirs(save_dir, exist_ok=True)

        epoch = self.training_state.current_epoch
        step = self.training_state.current_step_in_epoch
        global_step = self.training_state.current_global_step
        checkpoint_name = self._get_checkpoint_name(epoch, step, global_step)
        temp_checkpoint_dir = os.path.join(save_dir, f"tmp_{checkpoint_name}")
        final_checkpoint_dir = os.path.join(save_dir, checkpoint_name)
        if os.path.exists(final_checkpoint_dir):
            return

        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        try:
            # Save accelerator state (this includes model, optimizer, and scheduler)
            self.accelerator.save_state(temp_checkpoint_dir)

            # Save training metadata separately
            for base_name in ["training_state", "training_config", "log_config"]:
                file_path = os.path.join(temp_checkpoint_dir, f"{base_name}.pth")
                torch.save(getattr(self, base_name), file_path)

            os.rename(temp_checkpoint_dir, final_checkpoint_dir)
            self._cleanup_old_checkpoints(save_dir=save_dir)
            if self.accelerator.is_local_main_process:
                self.logger.info(f"Model saved to {final_checkpoint_dir}")

        except Exception as e:
            if self.accelerator.is_local_main_process:
                self.logger.error(f"Failed to save checkpoint: {e}")
            shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)
            raise

    def load(self) -> None:
        # check load condition ============================================================================
        checkpoint_dir = self.get_latest_checkpoint_dir()
        if len(os.listdir(checkpoint_dir)) <= 0:
            return

        # load ============================================================================================
        # Load accelerator state (this includes model, optimizer, and scheduler)
        self.accelerator.load_state(checkpoint_dir)

        # Load training metadata
        for base_name in ["training_state", "training_config", "log_config"]:
            file_path = os.path.join(checkpoint_dir, f"{base_name}.pth")
            if not os.path.exists(file_path):
                if self.accelerator.is_main_process:
                    self.logger.info(f"File {file_path} does not exist")
                continue
            setattr(self, base_name, torch.load(file_path, weights_only=False))

        if self.accelerator.is_main_process:
            self.logger.info(f"Model loaded from {checkpoint_dir}")
