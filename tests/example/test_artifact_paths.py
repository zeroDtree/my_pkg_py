"""Example configs keep artifacts relative to the directory the process starts in."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED = {
    "train.save_dir": "artifacts/checkpoints",
    "log.log_dir": "artifacts/logs",
    "mlflow.tracking_uri": "sqlite:///artifacts/mlflow.db",
    "mlflow.artifact_location": "artifacts/mlruns",
    "hydra.run.dir": "artifacts/logs",
    "hydra.sweep.dir": "artifacts/hydra",
}


def test_example_artifact_paths_stay_under_the_launch_directory():
    config_paths = sorted(
        path
        for path in (REPO_ROOT / "example").rglob("*.yaml")
        if "artifacts" not in path.parts
    )
    assert config_paths, "expected example configs"

    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        for key, expected in EXPECTED.items():
            value = OmegaConf.select(cfg, key, throw_on_missing=True)
            assert value == expected, f"{config_path.relative_to(REPO_ROOT)} {key} is {value!r}"
            assert not Path(str(value).removeprefix("sqlite:///")).is_absolute()
