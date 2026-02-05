import logging
from typing import Callable, List, Literal, Optional, cast

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from overrides import override
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

from ..util.decorators import inherit_docstrings
from ..util.iterator import inf_iterator
from .callback import BaseCallback, CallbackEvent
from .distributed_pipeline import DistributedPipeline, DistributedTrainingConfig, LogConfig
from .pipeline import LogConfig


@inherit_docstrings
class MyTrainingConfig(DistributedTrainingConfig):
    def __init__(
        self,
        train_strategy: Literal["epochs", "steps"] = "epochs",
        n_epochs: int = 100,
        batch_size: int = 16,
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
        n_steps: int | None = None,
        grad_clip_strategy: Literal["norm", "value", None] = "norm",
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        find_unused_parameters: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the MyTrainingConfig

        Args:
            train_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;], optional): the strategy determines whether to train the model for a fixed number of epochs or for a fixed number of steps. Defaults to "epochs".
            n_epochs (int, optional): the number of epochs. Defaults to 100.
            batch_size (int, optional): the batch size. Defaults to 16.
            device (str, optional): the device to use for training. Defaults to "cuda".
            save_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;, None], optional): the strategy determines whether to save the model and when to save it. Defaults to "epochs".
            save_dir (str | None, optional): the directory to save the model. Defaults to None.
            save_steps (int, optional): the number of steps to save the model. Defaults to 10.
            save_epochs (int, optional): the number of epochs to save the model. Defaults to 1.
            save_total_limit (int, optional): the maximum number of checkpoints to save. Defaults to 5.
            num_workers (int, optional): the number of workers to use for data loading. Defaults to 4.
            train_shuffle (bool, optional): whether to shuffle the training data. Defaults to True.
            eval_strategy (Literal[&quot;epochs&quot;, &quot;steps&quot;] | None, optional): the strategy determines whether to evaluate the model and when to evaluate it. Defaults to None.
            eval_steps (int, optional): the number of steps to evaluate the model. Defaults to 500.
            eval_epochs (int, optional): the number of epochs to evaluate the model. Defaults to 1.
            n_steps (int | None, optional): the number of steps to train the model. Defaults to None.
            grad_clip_strategy (Literal[&quot;norm&quot;, &quot;value&quot;, None], optional): the strategy determines whether to clip the gradient and how to clip it. Defaults to "norm".
            max_grad_norm (float, optional): the maximum gradient norm to clip the gradient. Defaults to 1.0.
            max_grad_value (float, optional): the maximum gradient value to clip the gradient. Defaults to 1.0.
            gradient_accumulation_steps (int, optional): the number of steps to accumulate gradients before updating the model. Defaults to 1.
            mixed_precision (str, optional): the mixed precision to use for training. Defaults to "fp16".
        """
        self.train_strategy = train_strategy
        real_batch_size = kwargs.get("real_batch_size", None)
        if real_batch_size is not None:
            print("real_batch_size is provided, so gradient_accumulation_steps is overridden")
            gradient_accumulation_steps = self.get_gradient_accumulation_steps(real_batch_size, batch_size)
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
            mixed_precision=mixed_precision,
            find_unused_parameters=find_unused_parameters,
        )
        self.n_steps: int | None = n_steps
        skip_keys = ["real_batch_size"]
        for key, value in kwargs.items():
            if key in skip_keys:
                continue
            setattr(self, key, value)

    def get_gradient_accumulation_steps(self, real_batch_size, per_device_batch_size) -> int:
        """Get the gradient accumulation steps

        Args:
            real_batch_size (int): the real batch size
            per_device_batch_size (int): the batch size per device

        Returns:
            int: the gradient accumulation steps
        """
        n_processes = Accelerator().num_processes
        assert real_batch_size % n_processes == 0, "real_batch_size must be divisible by n_processes"
        per_device_real_batch_size = real_batch_size // n_processes
        assert (
            per_device_real_batch_size % per_device_batch_size == 0
        ), "per_device_real_batch_size must be divisible by per_device_batch_size"
        gradient_accumulation_steps = per_device_real_batch_size // per_device_batch_size
        return gradient_accumulation_steps


@inherit_docstrings
class MyDistributedPipeline(DistributedPipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset | datasets.Dataset,
        eval_dataset: torch.utils.data.Dataset | datasets.Dataset,
        optimizers: tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR | torch.optim.lr_scheduler.CosineAnnealingLR
        ],
        training_config: MyTrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger | None,
        collate_fn: Callable | None = None,
        seed: int = 42,
        callbacks: Optional[List[BaseCallback]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            dataset=train_dataset,
            optimizers=optimizers,
            training_config=training_config,
            log_config=log_config,
            logger=logger,
            collate_fn=collate_fn,
            seed=seed,
            callbacks=callbacks,
            *args,
            **kwargs,
        )
        self.eval_dataset = eval_dataset
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            collate_fn=collate_fn,
        )
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        self.training_config = cast(MyTrainingConfig, self.training_config)

    @override
    def compute_loss(self, model, batch: dict) -> Tensor:
        self.trigger_callbacks(event=CallbackEvent.PRE_COMPUTE_LOSS, batch=batch)
        loss = None
        model_output = model(**batch)
        if isinstance(model_output, dict):
            assert "loss" in model_output, "model_output must contain 'loss' key if model_output is a dict"
            loss = model_output["loss"]
        else:
            loss = model_output
        self.trigger_callbacks(event=CallbackEvent.POST_COMPUTE_LOSS, atch=batch, loss=loss)
        return loss

    def eval_a_step(self, batch: dict) -> dict:
        """Evaluate the model for one step

        Args:
            batch (dict): the batch of data

        Returns:
            dict: a dictionary containing the evaluation loss
        """
        self.trigger_callbacks(event=CallbackEvent.PRE_EVAL_STEP, batch=batch)
        loss = self.compute_loss(self.model, batch).item()
        self.trigger_callbacks(event=CallbackEvent.POST_EVAL_STEP, batch=batch, loss=loss)
        return {"eval_loss": loss}

    @override
    def train(self):
        if self.training_config.train_strategy in ["epochs"]:
            return super().train()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_START)
        if self.training_config.n_steps is not None:
            self.training_set_iterator = inf_iterator(self.dataloader)
        else:
            raise ValueError("n_steps must be specified")
        i = 0
        result = None
        for _ in tqdm(range(self.training_config.n_steps), desc="training", mininterval=0):
            if i < self.training_state.current_global_step:
                i += 1
                continue
            batch = next(self.training_set_iterator)
            result = self.train_a_step(batch)

            if self._can_eval(flag="steps"):
                self.eval()
            if self._can_save(flag="steps"):
                self.save()
            self.training_state.current_step_in_epoch += 1
            self.training_state.current_global_step += 1
            i += 1
        self.save()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_END)
        return result

    @torch.no_grad()
    def eval(self, disable_grad: bool = False):
        self.trigger_callbacks(event=CallbackEvent.PRE_EVAL, eval_dataloader=self.eval_dataloader)
        self.model.eval()
        eval_results = []
        for batch in self.eval_dataloader:
            if disable_grad:
                with torch.no_grad():
                    result = self.eval_a_step(batch)
            else:
                result = self.eval_a_step(batch)
            eval_results.append(result)

        mean_eval_loss = sum([result["eval_loss"] for result in eval_results]) / len(eval_results)
        max_eval_loss = max([result["eval_loss"] for result in eval_results])
        min_eval_loss = min([result["eval_loss"] for result in eval_results])
        std_eval_loss = np.std([result["eval_loss"] for result in eval_results]).item()
        result = {
            "mean_eval_loss": mean_eval_loss,
            "max_eval_loss": max_eval_loss,
            "min_eval_loss": min_eval_loss,
            "std_eval_loss": std_eval_loss,
        }
        if self.accelerator.is_local_main_process:
            self.logger.info(f"[Testing] {result}")
            wandb.log(result, step=self.training_state.current_global_step)

        self.model.train()
        self.trigger_callbacks(event=CallbackEvent.POST_EVAL, eval_dataloader=self.eval_dataloader)
        return eval_results
