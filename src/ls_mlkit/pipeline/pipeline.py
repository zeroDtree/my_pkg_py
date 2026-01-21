import logging
import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import datasets
import torch
from torch import Tensor
from tqdm.auto import tqdm

import wandb

from .callback import BaseCallback, CallbackEvent, CallbackManager


class TrainingConfig:
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
        *args,
        **kwargs,
    ):
        """Initialize the TrainingConfig

        Args:
            n_epochs (int, optional): the number of epochs
            batch_size (int, optional): batch size. Defaults to 4.
            device (str, optional): device in training. Defaults to "cuda".
            save_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;, None], optional): save strategy determines whether to save the model and when to save it. Defaults to "epochs".
            save_dir (str, optional): directory to save the model. Defaults to None.
            save_steps (int, optional): save the model every n steps. Defaults to 10.
            save_epochs (int, optional): save the model every n epochs. Defaults to 1.
            save_total_limit (int, optional): maximum number of checkpoints to save. Defaults to 5.
            num_workers (int, optional): number of workers to use for data loading. Defaults to 4.
            train_shuffle (bool, optional): whether to shuffle the training data. Defaults to True.
            eval_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;], optional): evaluation strategy determines whether to evaluate the model and when to evaluate it. Defaults to None.
            eval_steps (int, optional): evaluate the model every n steps. Defaults to 500.
            eval_epochs (int, optional): evaluate the model every n epochs. Defaults to 1.
            grad_clip_strategy (Literal[&quot;norm&quot;, &quot;value&quot;, None], optional): gradient clip strategy determines whether to clip the gradient and how to clip it. Defaults to "norm".
            max_grad_norm (float, optional): maximum gradient norm. Defaults to 1.0.
            max_grad_value (float, optional): maximum gradient value. Defaults to 1.0.
            gradient_accumulation_steps (int, optional): number of steps to accumulate gradients before updating the model. Defaults to 1.

        Returns:
            None
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.save_strategy = save_strategy
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.save_epochs = save_epochs
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.save_total_limit = save_total_limit
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.eval_epochs = eval_epochs
        self.grad_clip_strategy = grad_clip_strategy
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.gradient_accumulation_steps = gradient_accumulation_steps


class LogConfig:
    def __init__(
        self,
        log_dir: str = "logs",
        log_steps: int = 100,
        log_epochs: int = 1,
        log_strategy: Literal["epochs", "steps"] = "epochs",
        *args,
        **kwargs,
    ):
        """Initialize the LogConfig

        Args:
            log_dir (str, optional): directory to save the logs. Defaults to "logs".
            log_steps (int, optional): log the metrics every n steps. Defaults to 100.
            log_epochs (int, optional): log the metrics every n epochs. Defaults to 1.
            log_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;], optional): log strategy determines whether to log the metrics and when to log them. Defaults to "epochs".
        """
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.log_epochs = log_epochs
        self.log_strategy = log_strategy


class TrainingState:
    def __init__(
        self,
        current_epoch: int = 0,
        current_step_in_epoch: int = 0,
        current_global_step: int = 0,
    ):
        """Initialize the TrainingState

        Args:
            current_epoch (int, optional): the current epoch. Defaults to 0.
            current_step_in_epoch (int, optional): the current step in the epoch. Defaults to 0.
            current_global_step (int, optional): the current global step. Defaults to 0.
        """
        self.current_epoch = current_epoch
        self.current_step_in_epoch = current_step_in_epoch
        self.current_global_step = current_global_step


class BasePipeline(metaclass=ABCMeta):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        training_config: TrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger,
        collate_fn: Optional[Callable] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        load_checkpoint: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the BasePipeline

        Args:
            model (torch.nn.Module): the model to train
            dataset (Union[torch.utils.data.Dataset, datasets.Dataset]): the dataset to train on
            optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]): the optimizers to use for training
            training_config (TrainingConfig): the training configuration
            log_config (LogConfig): the logging configuration
            logger (logging.Logger): the logger to use for logging
            collate_fn (Optional[Callable], optional): the collate function to use for the dataset. Defaults to None.
        """
        super().__init__()
        # callbacks
        self.callback_manager = CallbackManager()
        self.callback_manager.add_callbacks(callbacks=callbacks)

        # model, dataset, optimizers, log, training,
        self.model = model
        self.dataset = dataset
        self.optimizer, self.scheduler = optimizers
        self.training_config = training_config
        self.log_config = log_config
        self.training_state = TrainingState()
        self.logger = logger

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.training_config.batch_size,
            shuffle=self.training_config.train_shuffle,
            collate_fn=collate_fn,
            num_workers=self.training_config.num_workers,
        )

        if load_checkpoint and self.training_config.save_dir is not None and self.training_config.save_dir != "":
            self.load()
            self.training_config = training_config

    @abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: dict) -> Tensor | dict:
        """Compute the loss

        Args:
            model (torch.nn.Module): the model to train
            batch (dict): the batch of data

        Returns:
            Tensor | dict: loss Tensor or a dictionary containing the loss key
        """

    def train(self) -> None:
        """Train the model

        Returns:
            None
        """
        self.trigger_callbacks(event=CallbackEvent.TRAINING_START)
        n_epochs = self.training_config.n_epochs

        for epoch in tqdm(range(n_epochs), desc="training", mininterval=0):
            if epoch < self.training_state.current_epoch:
                continue
            self.train_an_epoch()
            self.training_state.current_epoch += 1

            if self._can_eval(flag="epochs"):
                self.eval()
            if self._can_save(flag="epochs"):
                self.save()
        self.save()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_END)

    def train_an_epoch(self):
        """Train the model for one epoch

        Returns:
            None
        """
        self.trigger_callbacks(event=CallbackEvent.EPOCH_START)
        for step, batch in enumerate(self.dataloader):
            if step < self.training_state.current_step_in_epoch:
                continue
            result = self.train_a_step(batch)

            if self._can_eval(flag="steps"):
                self.eval()
            if self._can_save(flag="steps"):
                self.save()
            self.training_state.current_step_in_epoch += 1
            self.training_state.current_global_step += 1

        if self._can_log(flag="epochs"):
            self.logger.info(
                f"[Training] Epoch {self.training_state.current_epoch}, Step {self.training_state.current_step_in_epoch}, Loss {result['loss']}"
            )
            wandb.log(result, step=self.training_state.current_epoch)
        self.training_state.current_step_in_epoch = 0
        self.trigger_callbacks(event=CallbackEvent.EPOCH_END)

    def train_a_step(self, batch: Dict[str, Any]) -> dict:
        """Train the model for one step

        Args:
            batch (Dict[str, Any]): the batch of data

        Returns:
            dict: a dictionary containing the loss key
        """
        self.trigger_callbacks(event=CallbackEvent.STEP_START, batch=batch)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        logger = self.logger
        device = self.training_config.device
        model.train()
        for key, value in batch.items():
            if type(value) == torch.Tensor:
                batch[key] = value.to(device)
        model = model.to(device)

        loss = self.compute_loss(model, batch)
        loss = loss / self.training_config.gradient_accumulation_steps
        self.trigger_callbacks(event=CallbackEvent.PRE_BACKWARD, loss=loss)
        loss.backward()
        self.trigger_callbacks(event=CallbackEvent.POST_BACKWARD, loss=loss)

        if ((self.training_state.current_global_step + 1) % self.training_config.gradient_accumulation_steps) == 0:
            self.gradient_clip()
            self.trigger_callbacks(event=CallbackEvent.PRE_OPTIMIZER_STEP)
            optimizer.step()
            self.trigger_callbacks(event=CallbackEvent.POST_OPTIMIZER_STEP)
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if self._can_log(flag="steps"):
            logger.info(
                f"[Training] Epoch {self.training_state.current_epoch}, Step {self.training_state.current_step_in_epoch}, Loss {loss.item()}"
            )
        self.trigger_callbacks(event=CallbackEvent.STEP_END, batch=batch)
        return {
            "loss": loss.item(),
        }

    def gradient_clip(self) -> None:
        """Clip the gradient

        Returns:
            None
        """
        model = self.model
        if self.training_config.grad_clip_strategy == "norm":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=self.training_config.max_grad_norm, norm_type=2, error_if_nonfinite=False
            )
        if self.training_config.grad_clip_strategy == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.training_config.max_grad_value)

    def eval(self):
        """Evaluate the model"""

    def _cleanup_old_checkpoints(self, save_dir):
        checkpoints = [d for d in os.listdir(save_dir) if d.startswith("checkpoint_")]
        checkpoints.sort(
            key=lambda x: int(re.search(r"global(\d+)", x).group(1)),
        )

        while len(checkpoints) > self.training_config.save_total_limit:
            oldest_checkpoint = checkpoints.pop(0)
            oldest_path = os.path.join(save_dir, oldest_checkpoint)
            shutil.rmtree(oldest_path)
            if self.logger is not None:
                self.logger.info(f"Deleted old checkpoint: {oldest_path}")

    def _get_checkpoint_name(self, epoch, step, global_step):
        return f"checkpoint_epoch{epoch}_step{step}_global{global_step}"

    def _can_save(self, flag: Literal["epochs", "steps"]):
        if (
            self.training_config.save_strategy is None
            or self.training_config.save_dir is None
            or self.training_config.save_dir == ""
        ):
            return False
        if flag == "epochs":
            return (self.training_state.current_epoch + 1) % self.training_config.save_epochs == 0
        elif flag == "steps":
            return (self.training_state.current_global_step + 1) % self.training_config.save_steps == 0
        else:
            return False

    def _can_log(self, flag: Literal["epochs", "steps"]):
        if self.logger is None:
            return False
        if self.log_config.log_strategy is None or self.log_config.log_dir is None or self.log_config.log_dir == "":
            return False
        if flag == "epochs":
            return (self.training_state.current_epoch + 1) % self.log_config.log_epochs == 0
        elif flag == "steps":
            return (self.training_state.current_global_step + 1) % self.log_config.log_steps == 0
        else:
            return False

    def _can_eval(self, flag: Literal["epochs", "steps"]):
        if self.training_config.eval_strategy is None:
            return False
        if flag == "epochs":
            return (self.training_state.current_epoch + 1) % self.training_config.eval_epochs == 0
        elif flag == "steps":
            return (self.training_state.current_global_step + 1) % self.training_config.eval_steps == 0
        else:
            return False

    def save(self) -> None:
        """Save the checkpoint

        Returns:
            None
        """
        self.trigger_callbacks(event=CallbackEvent.PRE_SAVE)
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
            for base_name in ["model", "optimizer", "scheduler", "training_state", "training_config", "log_config"]:
                file_path = os.path.join(temp_checkpoint_dir, f"{base_name}.pth")
                torch.save(getattr(self, base_name), file_path)
            os.rename(temp_checkpoint_dir, final_checkpoint_dir)
            self._cleanup_old_checkpoints(save_dir=save_dir)
            if self.logger is not None:
                self.logger.info(f"Model saved to {final_checkpoint_dir}")

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to save checkpoint: {e}")
            shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)
            raise
        self.trigger_callbacks(event=CallbackEvent.POST_SAVE)

    def load(self) -> None:
        """Load the checkpoint

        Returns:
            None
        """
        self.trigger_callbacks(event=CallbackEvent.PRE_LOAD)
        # check load condition ============================================================================
        checkpoint_dir = self.get_latest_checkpoint_dir()
        if checkpoint_dir is None or len(os.listdir(checkpoint_dir)) <= 0:
            return

        # load ============================================================================================

        for base_name in ["model", "optimizer", "scheduler", "training_state", "training_config", "log_config"]:
            file_path = os.path.join(checkpoint_dir, f"{base_name}.pth")
            if not os.path.exists(file_path):
                if self.logger is not None:
                    self.logger.info(f"File {file_path} does not exist")
                continue
            setattr(self, base_name, torch.load(file_path, weights_only=False))

        if self.logger is not None:
            self.logger.info(f"Model loaded from {checkpoint_dir}")
        self.trigger_callbacks(event=CallbackEvent.POST_LOAD)

    def get_latest_checkpoint_dir(self) -> str:
        save_dir = self.training_config.save_dir
        if save_dir is None or save_dir == "" or not os.path.exists(save_dir) or len(os.listdir(save_dir)) <= 0:
            return
        checkpoints = [d for d in os.listdir(save_dir) if d.startswith("checkpoint_")]
        checkpoints.sort(
            key=lambda x: int(re.search(r"global(\d+)", x).group(1)),
            reverse=True,
        )
        if len(checkpoints) <= 0:
            return
        checkpoint_dir = checkpoints.pop(0)
        checkpoint_dir = os.path.join(save_dir, checkpoint_dir)
        return checkpoint_dir

    def add_callbacks(self, callbacks: List[BaseCallback]):
        """Add a list of callbacks

        Args:
            callbacks (List[BaseCallback]): the callbacks to add
        """
        self.callback_manager.add_callbacks(callbacks=callbacks)

    def trigger_callbacks(self, event: CallbackEvent, *args, **kwargs):
        """Trigger all callbacks for a given event

        Args:
            event (CallbackEvent): the event to trigger
            *args: the arguments to pass to the callback
            **kwargs: the keyword arguments to pass to the callback
        """
        self.callback_manager.trigger(
            event=event,
            training_config=self.training_config,
            log_config=self.log_config,
            training_state=self.training_state,
            logger=self.logger,
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            *args,
            **kwargs,
        )
