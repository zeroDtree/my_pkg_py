import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.runner import run_experiment
from utils import get_collate_fn, get_dataset, get_model

from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = get_model(
        cfg,
        final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
    )["model"].model
    model = model.to(accelerator.device)

    result = model.sampling(
        shape=(16, 3, cfg.dataset.image_size, cfg.dataset.image_size),
        device=accelerator.device,
    )
    print(f"{result.keys()}")
    image = result["x"]
    print(f"Generated tensor shape: {image.shape}")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    save_dir = "fm_image"
    os.makedirs(save_dir, exist_ok=True)
    image_grid.save(os.path.join(save_dir, f"fm_uc_{cfg.flow.n_inference_steps}.png"))


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, get_model, get_dataset, get_collate_fn, post_train_fn=post_train)


if __name__ == "__main__":
    main()
