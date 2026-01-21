# official packages


from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
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
                **cfg.diffusion,
            },
        )

    # model
    result_get_model = get_model(cfg)
    model = result_get_model["model"]
    train_hook_handlers = result_get_model["train_hook_handlers"]
    sampling_hook_handlers = result_get_model["sampling_hook_handlers"]

    def get_callbacks(training_handlers):
        from callbacks import TrainingStepCallback

        return None

        return [TrainingStepCallback(training_handlers)]

    callbacks = get_callbacks(train_hook_handlers)

    for handler in train_hook_handlers:
        handler.disable()

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
        callbacks=callbacks,
    )

    pipeline.train()

    if accelerator.is_local_main_process:
        wandb.finish()

    if accelerator.is_local_main_process and True:
        import matplotlib.pyplot as plt
        import torch

        print(f"latest checkpoint dir = {pipeline.get_latest_checkpoint_dir()}")

        result_get_model = get_model(
            cfg,
            # model=unet_model,
            final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
        )

        model = result_get_model["model"]
        train_hook_handlers = result_get_model["train_hook_handlers"]
        sampling_hook_handlers = result_get_model["sampling_hook_handlers"]

        model = model.to(accelerator.device)
        n_samples = 256
        cfg.diffusion.n_discretization_steps
        n_inference_steps = cfg.diffusion.n_inference_steps

        for handler in sampling_hook_handlers:
            handler.disable()
        result: dict = model.sampling(shape=(n_samples, 2), device=accelerator.device, return_all=True)
        sample = result["x"]

        # Visualize sample
        sample_cpu = sample.detach().cpu()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(sample_cpu[:, 0], sample_cpu[:, 1], s=10, alpha=0.6)
        ax.set_title(f"Generated Samples (n={n_samples})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("sample_visualization.png", dpi=150)
        print(f"Sample visualization saved to 'sample_visualization.png'")
        print(f"Sample shape: {sample.shape}")
        print(f"Sample statistics - mean: {sample_cpu.mean(dim=0)}, std: {sample_cpu.std(dim=0)}")

        x_list = result["x_list"]
        print(f"total length of x_list={len(x_list)}")
        x_list = [x.detach().cpu() for x in x_list]
        len(x_list)

        n_steps = 8
        indexes = torch.linspace(0, n_inference_steps - 1, steps=n_steps + 1).round().long().numpy().tolist()
        print(f"paint indexes = {indexes}")
        x_list = [x_list[i] for i in indexes]

        fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)

        axes[0].scatter(x_list[0][:, 0], x_list[0][:, 1], s=10)
        axes[0].set_title(f"idx = {0}")
        axes[0].set_xlim(-3.0, 3.0)
        axes[0].set_ylim(-3.0, 3.0)

        for i in range(n_steps):
            x = x_list[i + 1]
            axes[i + 1].scatter(x[:, 0], x[:, 1], s=10)
            axes[i + 1].set_title(f"idx = {i+1}")

        plt.tight_layout()
        plt.savefig("sample_uc.png")

        # =================================

        for handler in sampling_hook_handlers:
            handler.enable()

        sigma = 1.0
        x = torch.randn(n_samples, 2) * sigma  # (n_samples, 2)

        # if you just want random labels –– otherwise load real labels here
        c_eval = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)  # (n_samples, 1)
        result: dict = model.sampling(
            shape=(256, 2), device=accelerator.device, return_all=True, sampling_condition=c_eval
        )
        x_list = result["x_list"]
        x_list = [x.detach().cpu() for x in x_list]

        n_steps = 8
        indexes = torch.linspace(0, n_inference_steps - 1, steps=n_steps + 1).round().long().numpy().tolist()
        print(f"paint indexes = {indexes}")
        x_list = [x_list[i] for i in indexes]

        # colours for the scatter (same length as x)
        colors = ["blue" if lbl == 0 else "orange" for lbl in c_eval.squeeze().tolist()]

        fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4), sharex=True, sharey=True)

        # initial frame
        axes[0].scatter(x[:, 0], x[:, 1], s=10, c=colors)
        axes[0].set_title(f"idx = 0")
        axes[0].set_xlim(-3.0, 3.0)
        axes[0].set_ylim(-3.0, 3.0)

        plot_count = 0
        with torch.no_grad():  # no gradients while sampling
            for i in range(n_steps):
                plot_count += 1
                x = x_list[i + 1]
                axes[plot_count].scatter(x[:, 0], x[:, 1], s=10, c=colors)
                axes[plot_count].set_title(f"idx = {i}")
                axes[plot_count].set_xlim(-3.0, 3.0)
                axes[plot_count].set_ylim(-3.0, 3.0)

        plt.tight_layout()
        plt.savefig("sample_c.png")

        # =================================
        # Visualize real dataset with categories for comparison
        from sklearn.datasets import make_moons

        n_real_samples = 1000
        x_real, c_real = make_moons(n_real_samples, noise=0.15)
        x_real = torch.tensor(x_real, dtype=torch.float32)
        c_real = torch.tensor(c_real, dtype=torch.long)

        # Create color map for categories
        colors_real = ["blue" if lbl == 0 else "orange" for lbl in c_real.tolist()]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(x_real[:, 0], x_real[:, 1], s=10, c=colors_real, alpha=0.6)
        ax.set_title(f"Real Dataset with Categories (n={n_real_samples})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor="blue", label="Class 0"), Patch(facecolor="orange", label="Class 1")]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        plt.savefig("real_dataset_with_categories.png", dpi=150)
        print(f"Real dataset visualization saved to 'real_dataset_with_categories.png'")
        print(f"Real dataset shape: {x_real.shape}")
        print(f"Class 0 count: {(c_real == 0).sum().item()}, Class 1 count: {(c_real == 1).sum().item()}")

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
                "lr": 1e-2,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
            },
            "train": {
                "seed": 97,
                "train_strategy": "steps",
                "n_epochs": 200,
                "n_steps": 1000,
                "num_workers": 1,
                "train_shuffle": True,
                "n_warmup_steps": 100,
                "lr_scheduler_type": "cosine_with_warmup",
                "batch_size": 16,
                "device": "cuda",
                "grad_clip_strategy": None,
                "max_grad_norm": 1.0,
                "max_grad_value": 1.0,
                "gradient_accumulation_steps": 4,
                "real_batch_size": 256,
                "save_strategy": "steps",  # can be "epochs", "steps", or null
                "save_dir": "checkpoints",
                "save_steps": 1000,
                "save_epochs": 100,
                "save_total_limit": 3,
                "eval_strategy": None,  # can be "epochs" or "steps" or null
                "eval_steps": 500,
                "eval_epochs": 1,
                "mixed_precision": None,
            },
            "log": {
                "log_dir": "logs",
                "log_steps": 10,
                "log_epochs": 1000,
                "log_strategy": "steps",
            },
            "wandb": {
                "reinit": True,
                "mode": "online",
                "project": "test",
                "group": "default",
                "entity": "superposed-tree",
            },
            "diffusion": {
                "name": "VPSDE",
                "n_discretization_steps": 50,
                "n_inference_steps": 50,
                "gs": 10.0,
            },
        }
    )
    import os
    import shutil

    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    # Accelerator(cpu=True, mixed_precision="fp16")

    for n_inference_steps in [1000]:
        cfg.diffusion.n_inference_steps = n_inference_steps
        cfg.diffusion.n_discretization_steps = n_inference_steps
        main(cfg)
