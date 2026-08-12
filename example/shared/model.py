"""Shared model building blocks for examples."""

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW


def mse_loss(predicted: Tensor, ground_truth: Tensor, mask: Tensor) -> Tensor:
    from torch.nn.functional import mse_loss as F_mse

    return F_mse(predicted, ground_truth)


def build_ttur_optimizer(generator: Module, discriminator: Module, cfg: DictConfig) -> AdamW:
    """One AdamW with separate G/D parameter groups (TTUR learning rates)."""
    g_cfg = cfg.optimizer.g
    d_cfg = cfg.optimizer.d
    weight_decay = float(cfg.optimizer.get("weight_decay", 0.0))
    return AdamW(
        [
            {"params": generator.parameters(), "lr": g_cfg.lr, "betas": tuple(g_cfg.betas)},
            {"params": discriminator.parameters(), "lr": d_cfg.lr, "betas": tuple(d_cfg.betas)},
        ],
        weight_decay=weight_decay,
    )


def build_gan_metrics_hook(*, log_interval: int):
    """Log d_loss / g_loss to wandb on POST_COMPUTE_LOSS (mirrors VAE metrics hook)."""
    import wandb

    from mlkit.util.hook.model_hook import ModelHook, ModelHookStageType

    state = {"step": 0}

    def fn(model, batch, model_output, **kwargs):
        state["step"] += 1
        if log_interval <= 0 or state["step"] % log_interval != 0:
            return
        stats: dict[str, float] = {"train/loss": float(model_output["loss"].item())}
        if "d_loss" in model_output:
            stats["train/d_loss"] = float(model_output["d_loss"].item())
        if "g_loss" in model_output:
            stats["train/g_loss"] = float(model_output["g_loss"].item())
        if "r1" in model_output:
            stats["train/r1"] = float(model_output["r1"].item())
        wandb.log(stats, step=state["step"])

    return ModelHook(name="gan_metrics", stage=ModelHookStageType.POST_COMPUTE_LOSS, fn=fn)


def build_unet2d(image_size: int) -> Any:
    """Build the standard UNet2DModel used across image diffusion/flow examples."""
    from diffusers import UNet2DModel

    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


class ImageModelWrapper(Module):
    """Wraps a UNet2DModel into the (**batch) -> dict[str, Tensor] interface."""

    def __init__(self, unet: Module):
        super().__init__()
        self.model = unet

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        x_t: Tensor = batch["x_t"]
        t: Tensor = batch["t"]
        return {"x": self.model(x_t, t, return_dict=False)[0]}


class MoonsMLP(Module):
    """4-layer MLP for 2-D moons data; accepts concatenated (t, x_t) input."""

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))


class MoonsModelWrapper(Module):
    """Wraps a MoonsMLP into the (**batch) -> dict[str, Tensor] interface."""

    def __init__(self, base_model: Module, unsqueeze_t: bool = True):
        super().__init__()
        self.model = base_model
        self.unsqueeze_t = unsqueeze_t

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        x_t: Tensor = batch["x_t"]
        t: Tensor = batch["t"]
        if self.unsqueeze_t:
            t = t.reshape(x_t.shape[0], 1)
        return {"x": self.model(x_t, t, return_dict=False)}


class ConditionalMoonsModelWrapper(Module):
    """Wraps ConditionalMoonsMLP into the (**batch) -> dict[str, Tensor] interface.

    Reads the class label from ``batch["c"]`` during training (key forwarded by
    ``RectifiedFlow.compute_loss``) and from ``batch["sampling_condition"]``
    during inference (key forwarded by ``RectifiedFlow.step``).
    """

    def __init__(self, base_model: Module):
        super().__init__()
        self.model = base_model

    def forward(self, **batch: Any) -> dict[str, Tensor]:
        x_t: Tensor = batch["x_t"]
        t: Tensor = batch["t"].unsqueeze(-1)
        c: Tensor = batch.get("c", batch.get("sampling_condition"))
        return {"x": self.model(x_t=x_t, t=t, c=c)}


class ConditionalMoonsMLP(Module):
    """4-layer MLP for conditional 2-D moons; accepts concatenated (t, c, x_t) input.

    Input width is ``dim + 2`` (1 time + 1 class label + dim data).
    """

    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            nn.Linear(dim + 2, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        return self.net(torch.cat((t, c, x_t), -1))


class MoonsClassifier(Module):
    """Simple classifier for the 2-D moons dataset."""

    def __init__(self, dim: int = 2, h: int = 64, n_labels: int = 2):
        super().__init__()
        from torch import nn

        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, n_labels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_moons_classifier(
    n_steps: int = 2000,
    noise: float = 0.15,
    batch_size: int = 256,
) -> MoonsClassifier:
    """Train a MoonsClassifier and print the final accuracy."""
    import torch.nn.functional as F
    from sklearn.datasets import make_moons
    from torch.optim import AdamW

    classifier = MoonsClassifier()
    optimizer = AdamW(classifier.parameters())

    for _ in range(n_steps):
        x, c = make_moons(batch_size, noise=noise)
        x_t = Tensor(x)
        c_t = Tensor(c).long()
        loss = F.cross_entropy(classifier(x_t), c_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x_eval, c_eval = make_moons(100, noise=noise)
    x_eval_t = Tensor(x_eval)
    c_eval_t = Tensor(c_eval).long()
    p_l = torch.argmax(classifier(x_eval_t), dim=-1)
    acc = (p_l == c_eval_t).float().mean()
    print(f"classifier acc={acc:.3f}")

    return classifier


def build_classifier_conditioner(
    base_conditioner_class: type,
    classifier_model: MoonsClassifier,
    guidance_scale: float,
) -> Any:
    """Factory: create a classifier-guidance conditioner extending *base_conditioner_class*.

    Works with both ``LGDConditioner`` (diffusion) and ``LGFMConditioner`` (flow matching).
    """
    import torch.nn.functional as F

    class ClassifierConditioner(base_conditioner_class):  # type: ignore[valid-type]  # ty: ignore[unsupported-base]
        def __init__(self) -> None:
            super().__init__(guidance_scale)
            self.classifier_model = classifier_model

        def prepare_condition_dict(self, train: bool = True, *args: Any, **kwargs: Any) -> dict:
            tgt_mask = kwargs.get("tgt_mask")
            assert tgt_mask is not None, "tgt_mask is required"
            posterior_mean_fn = kwargs.get("posterior_mean_fn")
            assert posterior_mean_fn is not None, "posterior_mean_fn is required"
            if train:
                gt_data = kwargs.get("gt_data")
                label = torch.argmax(self.classifier_model(gt_data), dim=-1)
            else:
                label = kwargs.get("sampling_condition")
                assert label is not None, "sampling_condition is required"
            return {"tgt_mask": tgt_mask, "label": label, "posterior_mean_fn": posterior_mean_fn}

        def set_condition(self, *args: Any, **kwargs: Any) -> None:
            self.tgt_mask = kwargs.get("tgt_mask")
            self.label = kwargs.get("label")
            self.posterior_mean_fn = kwargs.get("posterior_mean_fn")
            self.ready = True

        def compute_conditional_loss(self, p_gt_data: Tensor, padding_mask: Tensor) -> Tensor:
            assert self.label is not None
            c = self.label.squeeze(-1).long()
            return F.cross_entropy(self.classifier_model(p_gt_data), c)

    return ClassifierConditioner()


def get_device_from_accelerator() -> torch.device:
    """Return the current Accelerator device without keeping the object alive."""
    from transformers.trainer import Accelerator

    return Accelerator().device


_SuffixCallable = Callable[[], str]
