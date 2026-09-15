import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.runner import run_experiment
from utils import get_collate_fn, get_dataset, get_model


def post_train(cfg, model_result, pipeline, accelerator, train_set):
    import matplotlib.pyplot as plt
    import torch

    model = model_result["model"].model
    sampling_hook_handlers = model_result["sampling_hook_handlers"]
    model = model.to(accelerator.device)
    n_samples = 256

    def select_uniform_indices(total: int, k: int) -> list[int]:
        if total <= 0:
            return []
        k = min(k, total)
        if k == 1:
            return [0]
        return [int(round(i * (total - 1) / (k - 1))) for i in range(k)]

    # Unconditional sampling
    for handler in sampling_hook_handlers:
        handler.disable()
    result: dict = model.sampling(shape=(n_samples, 2), device=accelerator.device, return_all=True)
    x_list = [x.detach().cpu() for x in result["x_list"]]

    sel = select_uniform_indices(len(x_list), 8)
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4), sharex=True, sharey=True)
    if len(sel) == 1:
        axes = [axes]
    for ax, frame_idx in zip(axes, sel):
        x = x_list[frame_idx]
        ax.scatter(x[:, 0], x[:, 1], s=10)
        ax.set_title(f"idx = {frame_idx}")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
    fig.tight_layout()
    fig.savefig("sampling_unconditional.png")
    plt.close(fig)

    # Classifier-guided conditional sampling
    for handler in sampling_hook_handlers:
        handler.enable()
    c_eval = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)
    result_c: dict = model.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
        sampling_condition=c_eval,
    )
    x_list_c = [x.detach().cpu() for x in result_c["x_list"]]
    colors = ["blue" if lbl == 0 else "orange" for lbl in c_eval.squeeze().tolist()]

    sel = select_uniform_indices(len(x_list_c), 8)
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4), sharex=True, sharey=True)
    if len(sel) == 1:
        axes = [axes]
    for ax, frame_idx in zip(axes, sel):
        x = x_list_c[frame_idx]
        ax.scatter(x[:, 0], x[:, 1], s=10, c=colors)
        ax.set_title(f"idx = {frame_idx}")
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
    fig.tight_layout()
    fig.savefig("sampling_condition.png")
    plt.close(fig)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, get_model, get_dataset, get_collate_fn, post_train_fn=post_train, disable_train_hooks=True)


if __name__ == "__main__":
    main()
