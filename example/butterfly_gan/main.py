import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.model import build_ttur_optimizer
from shared.runner import run_experiment
from utils import get_collate_fn, get_dataset, get_model

from mlkit.generative_model.gan import GAN
from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def _tensor_to_pil_list(images):
    from diffusers.utils.pil_utils import numpy_to_pil

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    return numpy_to_pil(images)


def _to_pil_grid(images, rows: int = 4, cols: int = 4):
    from diffusers.utils.pil_utils import make_image_grid

    return make_image_grid(_tensor_to_pil_list(images), rows=rows, cols=cols)


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    import torch

    gan: GAN = model_result["model"].model
    gan = gan.to(accelerator.device)
    gan.eval()
    if gan.ema is not None:
        gan.ema.copy_to(gan.generator)

    image_size = cfg.dataset.image_size
    n_samples = 16

    with torch.no_grad():
        samples = gan.sampling(
            shape=(n_samples, 3, image_size, image_size),
            device=accelerator.device,
        )
        real_data = torch.stack(train_set[0:n_samples]["images"]).to(accelerator.device)

    latent_tag = f"latent:{cfg.gan.latent_dim}"
    sampling_path = f"sampling-{latent_tag}.png"
    comparison_path = f"comparison-{latent_tag}.png"

    _to_pil_grid(samples["x"]).save(sampling_path)
    print(f"Saved {sampling_path} (generated samples, EMA)")

    from diffusers.utils.pil_utils import make_image_grid

    real_and_generated = _tensor_to_pil_list(real_data) + _tensor_to_pil_list(samples["x"])
    make_image_grid(real_and_generated, rows=8, cols=4).save(comparison_path)
    print(f"Saved {comparison_path} (real on top, generated on bottom)")


def _get_optimizer(model, cfg):
    return build_ttur_optimizer(model.model.generator, model.model.discriminator, cfg)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(
        cfg,
        get_model,
        get_dataset,
        get_collate_fn,
        post_train_fn=post_train,
        get_optimizer_fn=_get_optimizer,
        save_dir_suffix=f"-latent:{cfg.gan.latent_dim}",
    )


if __name__ == "__main__":
    main()
