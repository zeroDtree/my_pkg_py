import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from omegaconf import DictConfig
from shared.runner import run_experiment
from torch import Tensor
from utils import get_collate_fn, get_dataset, get_model

from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = get_model(
        cfg,
        final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
    )["model"].model
    model = model.to(accelerator.device)

    x_0 = torch.stack(train_set[0:16]["images"]).to(accelerator.device)
    batch_size, channels, height, width = x_0.shape
    inpainting_mask = torch.zeros(batch_size, channels, height, width, device=accelerator.device)
    inpainting_mask[:, :, :, width // 2 :] = 1.0

    result: Tensor = model.inpainting(
        x=x_0,
        padding_mask=torch.ones(*x_0.shape, device=accelerator.device),
        inpainting_mask=inpainting_mask,
        device=accelerator.device,
        mode=cfg.diffuser.mode,
        n_repaint_steps=cfg.diffuser.n_repaint_steps,
    )
    image = (result / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    image_grid.save(
        f"inpainted_sample_{cfg.optimizer.name}_{cfg.diffuser.mode}_{cfg.diffuser.name}_{cfg.diffuser.n_inference_steps}_{cfg.diffuser.n_repaint_steps}.png"
    )


@hydra.main(config_path=".", config_name="config_inpainting", version_base=None)
def main(cfg: DictConfig):
    run_experiment(
        cfg,
        get_model,
        get_dataset,
        get_collate_fn,
        post_train_fn=post_train,
        save_dir_suffix=f"-{cfg.optimizer.name}-{cfg.diffuser.mode}",
    )


if __name__ == "__main__":
    main()
