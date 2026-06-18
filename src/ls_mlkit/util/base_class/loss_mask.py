"""Resolve coordinate loss masks via GM hooks."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base_gm_class import GMHookManager, GMHookStageType


def resolve_loss_mask(
    hook_manager: GMHookManager,
    *,
    padding_mask: Tensor,
    batch: dict[str, Any] | None,
) -> Tensor:
    """Return the atom mask used for ``loss_fn`` aggregation.

    Runs ``RESOLVE_LOSS_MASK`` hooks; falls back to ``padding_mask`` when no hook
    overrides or hooks return ``None``.
    """
    resolved = hook_manager.run_hooks(
        stage=GMHookStageType.RESOLVE_LOSS_MASK,
        tgt_key_name="loss_mask",
        loss_mask=padding_mask,
        padding_mask=padding_mask,
        batch=batch,
    )
    return padding_mask if resolved is None else resolved
