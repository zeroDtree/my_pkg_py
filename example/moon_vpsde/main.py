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
    n_inference_steps = cfg.diffusion.n_inference_steps
    n_frames = 8
    frame_indices = torch.linspace(0, n_inference_steps, steps=n_frames + 1).round().long().tolist()

    # Unconditional sampling
    for handler in sampling_hook_handlers:
        handler.disable()
    result: dict = model.sampling(shape=(n_samples, 2), device=accelerator.device, return_all=True)
    x_list = [x.detach().cpu() for x in result["x_list"]]
    x_list_sel = [x_list[i] for i in frame_indices]

    fig, axes = plt.subplots(1, n_frames + 1, figsize=(4 * (n_frames + 1), 4), sharex=True, sharey=True)
    for i, (ax, x) in enumerate(zip(axes, x_list_sel)):
        ax.scatter(x[:, 0], x[:, 1], s=10)
        ax.set_title(f"idx = {frame_indices[i]}")
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
    x_list_c_sel = [x_list_c[i] for i in frame_indices]
    colors = ["blue" if lbl == 0 else "orange" for lbl in c_eval.squeeze().tolist()]

    fig, axes = plt.subplots(1, n_frames + 1, figsize=(4 * (n_frames + 1), 4), sharex=True, sharey=True)
    for i, (ax, x) in enumerate(zip(axes, x_list_c_sel)):
        ax.scatter(x[:, 0], x[:, 1], s=10, c=colors)
        ax.set_title(f"idx = {frame_indices[i]}")
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
