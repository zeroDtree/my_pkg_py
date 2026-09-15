import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.runner import run_experiment
from utils_for_main import get_collate_fn, get_dataset, get_model

from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    import torch
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil
    from torch import Tensor

    from mlkit.diffusion.euclidean_edm_diffuser import EuclideanEDMDiffuser

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
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
        image_grid.save("samping_uc.png")

        E_x0_xt_list = result["E_x0_xt_list"]
        if E_x0_xt_list is not None and len(E_x0_xt_list) > 0:
            num_samples = min(8, len(E_x0_xt_list))
            indices = (
                [int(i * (len(E_x0_xt_list) - 1) / (num_samples - 1)) for i in range(num_samples)]
                if num_samples > 1
                else [0]
            )
            sampled_images = []
            for idx in indices:
                img = (E_x0_xt_list[idx][0:1] / 2 + 0.5).clamp(0, 1)
                sampled_images.extend(numpy_to_pil(img.cpu().permute(0, 2, 3, 1).numpy()))
            grid_rows = 2 if num_samples > 4 else 1
            grid_cols = (num_samples + grid_rows - 1) // grid_rows
            make_image_grid(sampled_images, rows=grid_rows, cols=grid_cols).save("denoising_process.png")
            print(f"Saved denoising process ({num_samples} of {len(E_x0_xt_list)} steps)")

    if cfg.inpainting:
        x_0 = torch.stack(train_set[0:16]["images"]).to(accelerator.device)
        batch_size, channels, height, width = x_0.shape
        inpainting_mask = torch.zeros(batch_size, channels, height, width, device=accelerator.device)
        inpainting_mask[:, :, :, width // 2 :] = 1.0

        result_inp: Tensor = model.inpainting(
            x=x_0,
            padding_mask=torch.ones(*x_0.shape, device=accelerator.device),
            inpainting_mask=inpainting_mask,
            device=accelerator.device,
            n_repaint_steps=cfg.gm.n_repaint_steps,
        )["x"]
        image = (result_inp / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        make_image_grid(numpy_to_pil(image), rows=4, cols=4).save(f"inpainted_sample_{cfg.gm.n_repaint_steps}.png")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, get_model, get_dataset, get_collate_fn, post_train_fn=post_train)


if __name__ == "__main__":
    main()
