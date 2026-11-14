"""VAE metrics stay inside the training step when the reporter fails."""

from __future__ import annotations

import pytest
import torch

from example.moon_vae.hooks import build_vae_metrics_hook
from mlkit.util.report import (
    clear_deferred_error,
    has_deferred_error,
    raise_deferred_error,
    use_reporter,
)


class _BoomReporter:
    def report(self, data, *, step=None) -> None:
        del data, step
        raise RuntimeError("backend down")


def test_vae_metrics_hook_defers_report_errors():
    hook = build_vae_metrics_hook(log_interval=1)
    values = torch.tensor([0.0, 1.0])
    model_output = {
        "mu": values,
        "logvar": values,
        "z": values,
        "x_hat": values,
        "recon_loss": torch.tensor(0.5),
        "kl_loss": torch.tensor(0.25),
        "loss": torch.tensor(0.75),
        "kl_weight": 1.0,
    }
    try:
        with use_reporter(_BoomReporter()):
            hook(model=None, batch={}, model_output=model_output)
        assert has_deferred_error()
        with pytest.raises(RuntimeError, match="backend down"):
            raise_deferred_error()
        assert not has_deferred_error()
    finally:
        clear_deferred_error()
