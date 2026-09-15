"""Regression test for epoch-based save/eval off-by-one.

After train_an_epoch() completes, current_epoch is already incremented before
_can_save/_can_eval(flag="epochs") are checked, so those helpers must NOT add +1.

Run with:
    uv run pytest tests/pipeline/test_epoch_off_by_one.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.pipeline import BasePipeline, LogConfig, TrainingConfig  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {"x": torch.tensor([float(index)]), "y": torch.tensor([float(index)])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x, y):
        pred = self.linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return {"loss": loss}


class TinyPipeline(BasePipeline):
    def compute_loss(self, model, batch):
        return model(**batch)


def make_pipeline(*, save_epochs: int = 2, eval_epochs: int = 2) -> TinyPipeline:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    config = TrainingConfig(
        n_epochs=4,
        batch_size=2,
        device="cpu",
        save_strategy="epochs",
        save_dir="/tmp/mlkit_epoch_off_by_one_test",
        save_epochs=save_epochs,
        eval_strategy="epochs",
        eval_epochs=eval_epochs,
        train_shuffle=False,
        num_workers=0,
    )
    return TinyPipeline(
        model=model,
        dataset=TinyDataset(),
        optimizers=(optimizer, scheduler),
        training_config=config,
        log_config=LogConfig(),
        logger=None,
        load_checkpoint=False,
    )


def test_can_save_completed_epoch_semantics():
    pipeline = make_pipeline(save_epochs=2)
    # current_epoch already equals completed epochs when _can_save("epochs") runs.
    pipeline.training_state.current_epoch = 0
    assert pipeline._can_save(flag="epochs") is False
    pipeline.training_state.current_epoch = 1
    assert pipeline._can_save(flag="epochs") is False
    pipeline.training_state.current_epoch = 2
    assert pipeline._can_save(flag="epochs") is True
    pipeline.training_state.current_epoch = 3
    assert pipeline._can_save(flag="epochs") is False
    pipeline.training_state.current_epoch = 4
    assert pipeline._can_save(flag="epochs") is True


def test_can_eval_completed_epoch_semantics():
    pipeline = make_pipeline(eval_epochs=2)
    pipeline.training_state.current_epoch = 0
    assert pipeline._can_eval(flag="epochs") is False
    pipeline.training_state.current_epoch = 1
    assert pipeline._can_eval(flag="epochs") is False
    pipeline.training_state.current_epoch = 2
    assert pipeline._can_eval(flag="epochs") is True
    pipeline.training_state.current_epoch = 3
    assert pipeline._can_eval(flag="epochs") is False
    pipeline.training_state.current_epoch = 4
    assert pipeline._can_eval(flag="epochs") is True
