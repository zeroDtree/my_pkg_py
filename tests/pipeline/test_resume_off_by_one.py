"""Regression test for mid-step checkpoint resume off-by-one.

Run with:
    uv run python tests/pipeline/test_resume_off_by_one.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from accelerate import skip_first_batches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.pipeline import BasePipeline, LogConfig, TrainingConfig  # noqa: E402


class CountingDataset:
    def __init__(self, n: int):
        self.n = n
        self.access_log: list[int] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        self.access_log.append(index)
        return {"x": torch.tensor([float(index)]), "y": torch.tensor([float(index)])}


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x, y):
        pred = self.linear(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return {"loss": loss}


def collate_fn(batch):
    xs = torch.stack([b["x"] for b in batch])
    ys = torch.stack([b["y"] for b in batch])
    return {"x": xs, "y": ys}


class SimplePipeline(BasePipeline):
    def compute_loss(self, model, batch):
        return model(**batch)


def make_pipeline(*, save_steps: int = 2) -> tuple[SimplePipeline, CountingDataset]:
    dataset = CountingDataset(20)  # 5 batches of size 4
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    config = TrainingConfig(
        n_epochs=1,
        batch_size=4,
        device="cpu",
        save_strategy="steps",
        save_dir="/tmp/mlkit_resume_off_by_one_test",
        save_steps=save_steps,
        train_shuffle=False,
        num_workers=0,
    )
    pipeline = SimplePipeline(
        model=model,
        dataset=dataset,
        optimizers=(optimizer, scheduler),
        training_config=config,
        log_config=LogConfig(),
        logger=None,
        collate_fn=collate_fn,
        load_checkpoint=False,
    )
    return pipeline, dataset


def test_can_save_completed_step_semantics():
    pipeline, _ = make_pipeline(save_steps=2)
    pipeline.training_state.current_global_step = 0
    assert pipeline._can_save(flag="steps") is False
    pipeline.training_state.current_global_step = 1
    assert pipeline._can_save(flag="steps") is False
    pipeline.training_state.current_global_step = 2
    assert pipeline._can_save(flag="steps") is True
    print("PASSED: _can_save completed-step semantics")


def test_resume_does_not_retrain_completed_batches():
    pipeline, dataset = make_pipeline(save_steps=2)

    # Train exactly two batches, then stop (simulates interrupt after a step save).
    it = iter(pipeline.dataloader)
    for _ in range(2):
        batch = next(it)
        pipeline.train_a_step(batch)
        pipeline.training_state.current_step_in_epoch += 1
        pipeline.training_state.current_global_step += 1

    assert pipeline.training_state.current_global_step == 2
    assert pipeline.training_state.current_step_in_epoch == 2
    assert pipeline._can_save(flag="steps") is True

    first_pass_samples = list(dataset.access_log)
    assert first_pass_samples == list(range(8)), f"expected samples 0..7, got {first_pass_samples}"

    # Resume mid-epoch: skip completed steps; must not re-fetch batches 0/1.
    dataset.access_log.clear()
    resume_step = pipeline.training_state.current_step_in_epoch
    active = skip_first_batches(pipeline.dataloader, resume_step)
    resumed_batch = next(iter(active))
    assert resumed_batch["x"][0].item() == 8.0
    assert dataset.access_log == [8, 9, 10, 11], f"unexpected resume access: {dataset.access_log}"
    assert set(range(8)).isdisjoint(dataset.access_log)

    print("PASSED: resume skips completed batches without re-training them")


if __name__ == "__main__":
    test_can_save_completed_step_semantics()
    test_resume_does_not_retrain_completed_batches()
    print("All resume off-by-one checks passed.")
