from ls_mlkit.util.huggingface import HF_MIRROR

HF_MIRROR.set_hf_mirror()
import os

from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from utils_for_main import (
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
                **cfg.gm,
            },
        )

    # model
    result_get_model = get_model(cfg)
    model = result_get_model["model"]
    # result_get_model["train_hook_handlers"]
    # sampling_hook_handlers = result_get_model["sampling_hook_handlers"]

    # for handler in train_hook_handlers:
    #     handler.disable()

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
        training_config.save_dir = get_new_save_dir(training_config.save_dir, cfg)

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

        from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil
        from ls_mlkit.diffusion.euclidean_edm_diffuser import EuclideanEDMDiffuser

        model: EuclideanEDMDiffuser = get_model(
            cfg,
            final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
        )["model"].model
        model = model.to(accelerator.device)
        print(type(model))

        if cfg.sampling:
            result: dict = model.sampling(
                shape=(16, 3, cfg.dataset.image_size, cfg.dataset.image_size),
                device=accelerator.device,
            )
            image = result["x"]
            print(f"Generated tensor shape: {image.shape}")
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()  # (batch_size, height, width, channels)

            image = numpy_to_pil(image)
            image_grid = make_image_grid(image, rows=4, cols=4)
            image_grid.save(f"samping_uc.png")

            E_x0_xt_list = result["E_x0_xt_list"]

            # Visualize E_x0_xt_list by uniformly sampling
            if E_x0_xt_list is not None and len(E_x0_xt_list) > 0:
                num_samples = min(8, len(E_x0_xt_list))  # Sample up to 8 timesteps
                indices = (
                    [int(i * (len(E_x0_xt_list) - 1) / (num_samples - 1)) for i in range(num_samples)]
                    if num_samples > 1
                    else [0]
                )

                sampled_images = []
                for idx in indices:
                    img_tensor = E_x0_xt_list[idx]
                    # Take only the first image from the batch
                    img_tensor = img_tensor[0:1]  # Shape: (1, 3, H, W)
                    # Normalize to [0, 1]
                    img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
                    img_tensor = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
                    sampled_images.extend(numpy_to_pil(img_tensor))

                # Create grid and save
                grid_rows = 2 if num_samples > 4 else 1
                grid_cols = (num_samples + grid_rows - 1) // grid_rows
                denoising_grid = make_image_grid(sampled_images, rows=grid_rows, cols=grid_cols)
                denoising_grid.save(f"denoising_process.png")
                print(
                    f"Saved denoising process visualization with {num_samples} timesteps from {len(E_x0_xt_list)} total steps"
                )

        if cfg.inpainting:
            import torch
            from torch import Tensor

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
                n_repaint_steps=cfg.gm.n_repaint_steps,
            )["x"]
            print(f"Inpainted tensor shape: {result.shape}")
            image = result
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()  # (batch_size, height, width, channels)
            image = numpy_to_pil(image)
            image_grid = make_image_grid(image, rows=4, cols=4)
            image_grid.save(f"inpainted_sample_{cfg.gm.n_repaint_steps}.png")
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
            "gm": {
                "name": "EuclideanEDMDiffuser",
                "n_discretization_steps": 256,
                "ndim_micro_shape": 3,
                "P_mean": -1.2,
                "P_std": 1.2,
                "sigma_data": 0.5,
                "sigma_min": 0.002,
                "sigma_max": 80.0,
                "rho": 7.0,
                "gs": 15.0,
                "n_repaint_steps": 3,
                "use_2nd_order_correction": True,
                "use_ode_flow": False,
                "use_clip": False,
                "clip_sample_range": 1.0,
                "use_dyn_thresholding": False,
                "dynamic_thresholding_ratio": 0.995,
                "sample_max_value": 1.0,
            },
            "sampling": False,
            "inpainting": True,
        }
    )
    import shutil

    # if os.path.exists("checkpoints"):
    #     shutil.rmtree("checkpoints")
    # Accelerator(cpu=True, mixed_precision="fp16")

    main(cfg)
