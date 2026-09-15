import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.runner import run_experiment
from utils import get_collate_fn, get_dataset, get_model

from mlkit.generative_model.ebm import EnergyBasedModel


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    import matplotlib.pyplot as plt

    model: EnergyBasedModel = model_result["model"].model
    model = model.to(accelerator.device)
    model.eval()

    n_samples = 512
    real_data = train_set.data[:n_samples].cpu().numpy()

    result = model.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
    )
    generated = result["x"].cpu().numpy()
    x_list = result["x_list"]

    def select_uniform_indices(total: int, k: int) -> list[int]:
        if total <= 0:
            return []
        k = min(k, total)
        if k == 1:
            return [0]
        return [int(round(i * (total - 1) / (k - 1))) for i in range(k)]

    # Real vs final Langevin samples
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for ax, data, title in zip(
        axes,
        [real_data, generated],
        ["Real", "Langevin samples"],
        strict=True,
    ):
        ax.scatter(data[:, 0], data[:, 1], s=8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"sampling-sigma:{cfg.ebm.sigma}.png")
    plt.close(fig)
    print(f"Saved sampling-sigma:{cfg.ebm.sigma}.png")

    # Trajectory frames
    sel = select_uniform_indices(len(x_list), 8)
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4), sharex=True, sharey=True)
    if len(sel) == 1:
        axes = [axes]
    for ax, frame_idx in zip(axes, sel, strict=True):
        x = x_list[frame_idx]
        ax.scatter(x[:, 0], x[:, 1], s=10, alpha=0.6)
        ax.set_title(f"step = {frame_idx}")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"trajectory-sigma:{cfg.ebm.sigma}.png")
    plt.close(fig)
    print(f"Saved trajectory-sigma:{cfg.ebm.sigma}.png")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(
        cfg,
        get_model,
        get_dataset,
        get_collate_fn,
        post_train_fn=post_train,
        save_dir_suffix=f"-sigma:{cfg.ebm.sigma}",
    )


if __name__ == "__main__":
    main()
