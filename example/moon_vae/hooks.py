"""External ModelHooks for moon_vae: KL annealing and debug logging."""

from __future__ import annotations

from typing import Any

import wandb
from torch import Tensor

from mlkit.util.hook.model_hook import ModelHook, ModelHookStageType


def vae_debug_stats(model_output: dict[str, Any], *, kl_weight: float) -> dict[str, float]:
    mu: Tensor = model_output["mu"]
    logvar: Tensor = model_output["logvar"]
    z: Tensor = model_output["z"]
    x_hat: Tensor = model_output["x_hat"]
    return {
        "debug/mu_std": mu.std().item(),
        "debug/logvar_mean": logvar.mean().item(),
        "debug/z_std": z.std().item(),
        "debug/x_hat_std": x_hat.std().item(),
        "debug/kl_weight": kl_weight,
        "train/recon_loss": model_output["recon_loss"].item(),
        "train/kl_loss": model_output["kl_loss"].item(),
    }


def format_vae_train_line(step: int, model_output: dict[str, Any], stats: dict[str, float]) -> str:
    loss = model_output["loss"].item()
    return (
        f"[VAE step={step}] "
        f"loss={loss:.4f} "
        f"recon={stats['train/recon_loss']:.4f} "
        f"kl={stats['train/kl_loss']:.4f} "
        f"kl_weight={stats['debug/kl_weight']:.4f} "
        f"mu_std={stats['debug/mu_std']:.4f} "
        f"logvar_mean={stats['debug/logvar_mean']:.4f} "
        f"z_std={stats['debug/z_std']:.4f} "
        f"x_hat_std={stats['debug/x_hat_std']:.4f}"
    )


def build_kl_anneal_hook(*, target: float, anneal_steps: int) -> ModelHook:
    state = {"step": 0}

    def fn(model, batch, **kwargs):
        if anneal_steps > 0:
            t = min(1.0, state["step"] / anneal_steps)
        else:
            t = 1.0
        batch["kl_weight"] = target * t
        state["step"] += 1

    return ModelHook(name="kl_anneal", stage=ModelHookStageType.PRE_COMPUTE_LOSS, fn=fn)


def build_vae_metrics_hook(*, log_interval: int) -> ModelHook:
    state = {"step": 0}

    def fn(model, batch, model_output, **kwargs):
        state["step"] += 1
        if log_interval <= 0 or state["step"] % log_interval != 0:
            return
        kl_weight = model_output.get("kl_weight", 1.0)
        if isinstance(kl_weight, Tensor):
            kl_weight = kl_weight.item()
        stats = vae_debug_stats(model_output, kl_weight=float(kl_weight))
        stats["train/loss"] = model_output["loss"].item()
        wandb.log(stats, step=state["step"])
        print(format_vae_train_line(state["step"], model_output, stats), flush=True)

    return ModelHook(name="vae_metrics", stage=ModelHookStageType.POST_COMPUTE_LOSS, fn=fn)


build_vae_log_hook = build_vae_metrics_hook
