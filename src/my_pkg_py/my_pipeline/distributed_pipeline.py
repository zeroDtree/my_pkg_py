from typing import Callable, Optional, Tuple, Union
from my_pipeline.pipeline import BasePipeline, LogConfig, TrainingConfig
import torch
from tqdm import tqdm
import datasets
from accelerate import Accelerator
import accelerate
from overrides import override
import os
import re
import shutil
from typing import Dict, Any
from my_utils import Observer, wandb_logger
import logging
import wandb


class DistributedPipeline(BasePipeline):
    @override
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        training_config: TrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger,
        collate_fn: Optional[Callable] = None,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        accelerate.utils.set_seed(seed)

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
        # Initialize accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=training_config.gradient_accumulation_steps)

        # Prepare everything for distributed training
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        if self.accelerator.is_main_process:
            assert self.logger is not None, f"Error from {self.__class__.__name__}: logger is required"
            self.logger.info(f"Using distributed training with accelerate")
            self.logger.info(f"Number of processes: {self.accelerator.num_processes}")
            self.logger.info(f"Current device: {self.accelerator.device}")

    @override
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
        if self.accelerator.is_local_main_process:
            wandb.log(result)

        return result

    @override
    def save(self):
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

    @override
    def load(self):
        # check load condition ============================================================================
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
            setattr(self, base_name, torch.load(file_path))

        if self.accelerator.is_main_process:
            self.logger.info(f"Model loaded from {checkpoint_dir}")
