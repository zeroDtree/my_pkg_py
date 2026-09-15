"""Regression test: DistributedPipeline.save must call save_state on all ranks.

FSDP/DeepSpeed use collectives inside accelerator.save_state, so non-main ranks
must not early-return before that call.

Run with:
    uv run pytest tests/pipeline/test_distributed_save_all_ranks.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.distributed_pipeline import (  # noqa: E402
    DistributedPipeline,
    DistributedTrainingConfig,
)
from mlkit.pipeline.pipeline import LogConfig, TrainingState  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {"x": torch.tensor([float(index)])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class TinyDistributedPipeline(DistributedPipeline):
    def compute_loss(self, model, batch):
        pred = model(batch["x"])
        return pred.sum()


def _make_stub_accelerator(*, is_main_process: bool) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_main_process = is_main_process
    accelerator.is_local_main_process = is_main_process
    accelerator.num_processes = 2
    accelerator.device = torch.device("cpu")
    accelerator.prepare.side_effect = lambda *args: args
    accelerator.wait_for_everyone = MagicMock()
    accelerator.save_state = MagicMock()
    return accelerator


def test_save_calls_save_state_on_non_main_process(monkeypatch):
    accelerator = _make_stub_accelerator(is_main_process=False)

    # Bypass real Accelerator construction in DistributedPipeline.__init__.
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.Accelerator",
        lambda *args, **kwargs: accelerator,
    )
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.accelerate.utils.set_seed",
        lambda *args, **kwargs: None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=2,
            device="cpu",
            save_strategy="steps",
            save_dir=tmp,
            save_steps=1,
            num_workers=0,
            train_shuffle=False,
            mixed_precision="no",
        )
        pipeline = TinyDistributedPipeline(
            model=model,
            dataset=TinyDataset(),
            optimizers=(optimizer, scheduler),
            training_config=config,
            log_config=LogConfig(log_dir=""),
            logger=None,
        )
        # Re-attach the stub after __init__ (prepare may replace references).
        pipeline.accelerator = accelerator
        pipeline.training_state = TrainingState(current_epoch=0, current_step_in_epoch=1, current_global_step=1)

        pipeline.save()

        accelerator.save_state.assert_called_once()
        accelerator.wait_for_everyone.assert_called()
        # Non-main process must not rename into a final checkpoint directory.
        checkpoint_dirs = list(Path(tmp).glob("checkpoint_*"))
        assert checkpoint_dirs == []
