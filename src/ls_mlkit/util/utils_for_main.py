import math
import os

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.nn import Module


def get_run_name(cfg: DictConfig, prefix: str = "", suffix: str = "") -> str:
    run_name = f"{cfg.dataset.name}-{cfg.model.name.replace('/','_').replace('-','_')}-lr:{cfg.optimizer.lr}"
    if cfg.train.train_strategy in ["steps"]:
        run_name += f"-n_steps:{cfg.train.n_steps}"
    if cfg.train.train_strategy in ["epochs"]:
        run_name += f"-n_epochs:{cfg.train.n_epochs}"
    run_name = f"{prefix}{run_name}{suffix}"
    return run_name


def get_optimizer(model, cfg: DictConfig):
    import torch  # type: ignore

    optimizer_config = dict(cfg.optimizer)
    optimizer_class_name = optimizer_config.pop("name")
    optimizer = eval("torch.optim." + optimizer_class_name)(model.parameters(), **optimizer_config)
    return optimizer


def get_learing_rate_scheduler(optimizer, accelerator: Accelerator, train_set, cfg: DictConfig, inplace: bool = True):
    from ls_mlkit.scheduler.lr_scheduler_factory import get_lr_scheduler

    cfg.train.batch_size
    real_batch_size = cfg.train.get(
        "real_batch_size", cfg.train.batch_size * cfg.train.gradient_accumulation_steps * accelerator.num_processes
    )
    effective_batch_size = accelerator.num_processes * cfg.train.batch_size
    assert real_batch_size % effective_batch_size == 0, "real_batch_size must be divisible by effective_batch_size"
    gradient_accumulation_steps = real_batch_size // effective_batch_size

    if cfg.train.train_strategy in ["epochs"]:
        n_training_steps = math.ceil(1.0 * len(train_set) * cfg.train.n_epochs / effective_batch_size)
    elif cfg.train.train_strategy in ["steps"]:
        n_training_steps = cfg.train.n_steps
    else:
        raise ValueError(f"Train Strategy {cfg.train.train_strategy} is not supported")
    if inplace:
        cfg.train.n_steps = n_training_steps
        cfg.train.real_batch_size = real_batch_size
        cfg.train.gradient_accumulation_steps = gradient_accumulation_steps

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        n_warmup_steps=cfg.train.n_warmup_steps,
        n_training_steps=(n_training_steps * accelerator.num_processes),
        lr_scheduler_type=cfg.train.lr_scheduler_type,
    )
    return lr_scheduler


def get_train_class():
    from ls_mlkit.pipeline.dist_pipeline_impl import MyDistributedPipeline, MyTrainingConfig

    return MyDistributedPipeline, MyTrainingConfig


def get_new_save_dir(save_dir, cfg: DictConfig, prefix: str = "", suffix: str = ""):
    if save_dir is None:
        save_dir = "checkpoints"
    model_name = str(cfg.model.name.replace("/", "_").replace("-", "_"))
    dataset_name = str(cfg.dataset.id)
    lr = str(cfg.optimizer.lr)
    batch_size = str(cfg.train.batch_size)
    new_save_dir = os.path.join(save_dir, f"{model_name}/{dataset_name}/-lr:{lr}-bs:{batch_size}")
    new_save_dir = f"{prefix}{new_save_dir}{suffix}"
    return new_save_dir


def load_checkpoint(model: Module, final_model_ckpt_path: str) -> Module:
    # Handle different checkpoint formats
    if final_model_ckpt_path.endswith(".safetensors"):
        print(f"Loading safetensors checkpoint: {final_model_ckpt_path}")
        try:
            from safetensors.torch import load_file

            checkpoint = load_file(final_model_ckpt_path)
            # For safetensors, the state_dict is directly the checkpoint
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from safetensors: {final_model_ckpt_path}")
        except ImportError:
            print("safetensors library not found. Install with: pip install safetensors")
            raise
        except Exception as e:
            print(f"Error loading safetensors file: {e}")
            print("Trying to load model from the checkpoint...")
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model from safetensors: {final_model_ckpt_path}")
    else:
        print(f"Loading PyTorch checkpoint: {final_model_ckpt_path}")
        checkpoint = torch.load(final_model_ckpt_path, map_location="cpu", weights_only=False)
        # print(f"Loaded PyTorch checkpoint with keys: {list(checkpoint.keys())}")
        if "model" in checkpoint:
            # If saved via pipeline, the diffuser is under 'model' key
            model.load_state_dict(checkpoint["model"])
        else:
            # If saved directly as state_dict
            model.load_state_dict(checkpoint)
        print(f"Loaded model from PyTorch checkpoint: {final_model_ckpt_path}")
    return model
