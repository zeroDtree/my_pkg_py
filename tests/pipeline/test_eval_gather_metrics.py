"""Regression test: eval metrics must aggregate losses across process shards.

The multi-process spawn path is covered indirectly by mocking
accelerator.gather_for_metrics to concatenate two shards (the same API
MyDistributedPipeline.eval relies on). A real 2-process hang-prone spawn
is intentionally avoided in CI.

Run with:
    uv run pytest tests/pipeline/test_eval_gather_metrics.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.dist_pipeline_impl import MyDistributedPipeline, MyTrainingConfig  # noqa: E402
from mlkit.pipeline.pipeline import LogConfig  # noqa: E402


class ValueDataset(Dataset):
    def __init__(self, values: list[float]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        v = float(self.values[index])
        return {"x": torch.tensor([v]), "y": torch.tensor([v])}


class ZeroModel(torch.nn.Module):
    """Predicts zeros so MSE(pred, y) == mean(y^2)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()

    def forward(self, x, y):
        pred = self.linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return {"loss": loss}


def collate_fn(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


def _make_pipeline(values: list[float]) -> MyDistributedPipeline:
    model = ZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    logger = logging.getLogger("eval_gather_metrics")
    logger.addHandler(logging.NullHandler())
    config = MyTrainingConfig(
        train_strategy="epochs",
        n_epochs=1,
        batch_size=2,
        device="cpu",
        save_strategy=None,
        save_dir="",
        num_workers=0,
        train_shuffle=False,
        mixed_precision="no",
    )
    return MyDistributedPipeline(
        model=model,
        train_dataset=ValueDataset(values),
        eval_dataset=ValueDataset(values),
        optimizers=(optimizer, scheduler),
        training_config=config,
        log_config=LogConfig(log_dir=""),
        logger=logger,
        collate_fn=collate_fn,
        seed=42,
    )


def test_eval_calls_gather_for_metrics(monkeypatch):
    values = [1.0, 2.0, 3.0, 4.0]
    pipeline = _make_pipeline(values)

    calls = {"n": 0}
    original = pipeline.accelerator.gather_for_metrics

    def tracking_gather(x, *args, **kwargs):
        calls["n"] += 1
        return original(x, *args, **kwargs)

    monkeypatch.setattr(pipeline.accelerator, "gather_for_metrics", tracking_gather)
    import mlkit.pipeline.dist_pipeline_impl as impl

    monkeypatch.setattr(impl.wandb, "log", lambda *a, **k: None)
    pipeline.eval()
    assert calls["n"] > 0, "eval() must call accelerator.gather_for_metrics"


def test_eval_aggregates_across_simulated_shards(monkeypatch):
    """Simulate a second process shard via mocked gather_for_metrics.

    Local dataloader yields batches from values A; gather concatenates a fake
    shard from values B so summary stats must reflect A∪B, not A alone.
    """
    local_values = [1.0, 2.0, 3.0, 4.0]
    pipeline = _make_pipeline(local_values)

    # Precompute the other-rank batch-mean losses with the same batch_size=2
    # and replicate by batch size, matching eval()'s gather input shape.
    other_batches = [
        torch.tensor([5.0, 6.0]),
        torch.tensor([7.0, 8.0]),
    ]
    other_loss_tensors = []
    for y in other_batches:
        batch_mean = float((y * y).mean().item())  # zero-model MSE
        other_loss_tensors.append(torch.tensor(batch_mean).repeat(y.numel()))
    other_iter = iter(other_loss_tensors)

    original_gather = pipeline.accelerator.gather_for_metrics

    def fake_gather(local_tensor, *args, **kwargs):
        # In single-process, original gather is identity; append the other shard.
        local = original_gather(local_tensor, *args, **kwargs)
        other = next(other_iter)
        return torch.cat([local.detach().cpu(), other.detach().cpu()])

    monkeypatch.setattr(pipeline.accelerator, "gather_for_metrics", fake_gather)

    logged = {}

    def fake_wandb_log(payload, step=None):
        logged.update(payload)

    import mlkit.pipeline.dist_pipeline_impl as impl

    monkeypatch.setattr(impl.wandb, "log", fake_wandb_log)

    pipeline.eval()

    # Expected: all batch-mean losses from local + other, each repeated batch_size times.
    # Local batches: [1,2] mean=2.5, [3,4] mean=12.5  (y^2 means: (1+4)/2=2.5, (9+16)/2=12.5)
    # Other batches: [5,6] mean=30.5, [7,8] mean=56.5
    expected = torch.tensor([2.5, 2.5, 12.5, 12.5, 30.5, 30.5, 56.5, 56.5])
    assert "mean_eval_loss" in logged
    assert abs(logged["mean_eval_loss"] - expected.mean().item()) < 1e-5
    assert abs(logged["max_eval_loss"] - expected.max().item()) < 1e-5
    assert abs(logged["min_eval_loss"] - expected.min().item()) < 1e-5

    # Local-only mean would be wrong (would be ~7.5); ensure we didn't fall back to that.
    local_only = torch.tensor([2.5, 2.5, 12.5, 12.5])
    assert abs(logged["mean_eval_loss"] - local_only.mean().item()) > 1.0
