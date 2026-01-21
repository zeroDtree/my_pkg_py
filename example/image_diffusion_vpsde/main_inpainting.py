# official packages

import torch
from accelerate import Accelerator
from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from utils import (
    get_collate_fn,
    get_dataset,
    get_learing_rate_scheduler,
    get_model,
    get_new_save_dir,
    get_optimizer,
    get_run_name,
    get_train_class,
)

import wandb
from ls_mlkit.pipeline.pipeline import LogConfig
from ls_mlkit.util.log import get_and_create_new_log_dir, get_logger
from ls_mlkit.util.seed import seed_everything
from ls_mlkit.util.show import show_info


def main(cfg: DictConfig):
    # seed
    seed_everything(cfg.train.seed)

    # distributed training
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    print(f"accelerator.device = {accelerator.device}")

    # logger
    if accelerator.is_local_main_process:
        log_dir = get_and_create_new_log_dir(cfg.log.log_dir)
        logger = get_logger(name=str(main.__name__), log_dir=log_dir)
        logger.info(f"accelerator.device = {accelerator.device}")
        logger.info(f"seed = {cfg.train.seed}")

    # wandb
    run_name = get_run_name(cfg)

    if accelerator.is_local_main_process:
        logger.info("Config: \n" + OmegaConf.to_yaml(cfg))  # type: ignore
        wandb.init(
            reinit=cfg.wandb.reinit,
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            name=run_name,
            group=cfg.wandb.group,
            entity=cfg.wandb.entity,
            config={
                **cfg.dataset,
                **cfg.model,
                **cfg.optimizer,
                **cfg.train,
                **cfg.log,
                **cfg.diffuser,
            },
        )

    # model
    model = get_model(cfg)

    # dataset
    train_set, val_set, test_set = get_dataset(cfg)

    # optimiers
    optimizer = get_optimizer(model, cfg)

    # lr scheduler
    lr_scheduler = get_learing_rate_scheduler(optimizer, accelerator, train_set, cfg)
    show_info(model=model, optimizer=optimizer)

    # pipeline
    log_config = LogConfig(**cfg.log)
    PipelineClass, TrainingConfigClass = get_train_class()
    training_config = TrainingConfigClass(**cfg.train)
    if accelerator.is_local_main_process:
        training_config.save_dir = get_new_save_dir(
            training_config.save_dir, cfg, suffix=f"-{cfg.optimizer.name}-{cfg.diffuser.mode}"
        )

    print(training_config.__dict__)

    pipeline = PipelineClass(
        model=model,
        train_dataset=train_set,
        eval_dataset=val_set,
        optimizers=(optimizer, lr_scheduler),
        training_config=training_config,
        log_config=log_config,
        collate_fn=get_collate_fn(cfg),  # Pass cfg to get_collate_fn
        logger=logger if accelerator.is_local_main_process else None,
    )

    pipeline.train()

    if accelerator.is_local_main_process:
        wandb.finish()

    if accelerator.is_local_main_process:

        model = get_model(
            cfg,
            # model=unet_model,
            final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
        )
        model = model.to(accelerator.device)

        x_0 = torch.stack(train_set[0:16]["images"]).to(accelerator.device)
        print(f"x_0 shape: {x_0.shape}")

        # Create inpainting mask to remove right half of the image
        # inpainting_mask: 1 = inpaint (remove), 0 = keep original
        batch_size, channels, height, width = x_0.shape
        inpainting_mask = torch.zeros(batch_size, channels, height, width, device=accelerator.device)
        # Set right half to 1 (will be inpainted/removed)
        inpainting_mask[:, :, :, width // 2 :] = 1.0

        result: Tensor = model.inpainting(
            x=x_0,
            padding_mask=torch.ones(*x_0.shape, device=accelerator.device),
            inpainting_mask=inpainting_mask,
            device=accelerator.device,
            mode=cfg.diffuser.mode,
            n_repaint_steps=cfg.diffuser.n_repaint_steps,
        )
        image = result
        print(f"Inpainted tensor shape: {result.shape}")
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()  # (batch_size, height, width, channels)

        image = numpy_to_pil(image)
        image_grid = make_image_grid(image, rows=4, cols=4)
        image_grid.save(
            f"inpainted_sample_{cfg.optimizer.name}_{cfg.diffuser.mode}_{cfg.diffuser.name}_{cfg.diffuser.n_inference_steps}_{cfg.diffuser.n_repaint_steps}.png"
        )

    return


if __name__ == "__main__":
    cfg = DictConfig(
        {
            "model": {
                "name": "UNet2DModel",
            },
            "dataset": {
                "id": "huggan/smithsonian_butterflies_subset",
                "name": "smithsonian_butterflies_subset",
                "image_size": 128,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
            },
            "train": {
                "seed": 97,
                "train_strategy": "epochs",
                "n_epochs": 50,
                "n_steps": 40388,
                "num_workers": 1,
                "train_shuffle": True,
                "n_warmup_steps": 500,
                "lr_scheduler_type": "cosine_with_warmup",
                "batch_size": 16,
                "device": "cuda",
                "grad_clip_strategy": "norm",
                "max_grad_norm": 1.0,
                "max_grad_value": 1.0,
                "gradient_accumulation_steps": 1,
                "real_batch_size": 16,
                "save_strategy": "steps",  # can be "epochs", "steps", or null
                "save_dir": "checkpoints",
                "save_steps": 1000,
                "save_epochs": 1,
                "save_total_limit": 3,
                "eval_strategy": None,  # can be "epochs" or "steps" or null
                "eval_steps": 500,
                "eval_epochs": 1,
                "mixed_precision": "fp16",
            },
            "log": {
                "log_dir": "logs",
                "log_steps": 1,
                "log_epochs": 1,
                "log_strategy": "steps",
            },
            "wandb": {
                "reinit": True,
                "mode": "offline",
                "project": "test",
                "group": "default",
                "entity": "superposed-tree",
            },
            "diffuser": {
                "name": "DDPM",
                "n_discretization_steps": 1000,
                "n_inference_steps": 20,
                "eta": 0.0,
                "mode": "epsilon",
                "recovery_mode": "x_t",
                "n_repaint_steps": 1,
            },
        }
    )
    for optimizer_name in ["AdamW"]:
        for diffusion_mode in ["epsilon"]:
            for n_inference_steps in [1000]:
                for n_repaint_steps in [1, 3]:
                    cfg.optimizer.name = optimizer_name
                    cfg.diffuser.mode = diffusion_mode
                    cfg.diffuser.n_inference_steps = n_inference_steps
                    cfg.diffuser.n_repaint_steps = n_repaint_steps
                    main(cfg)
