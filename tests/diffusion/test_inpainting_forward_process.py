"""Tests for the unified two-time forward_process contract used by inpainting."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mlkit.diffusion.euclidean_ddpm_diffuser import EuclideanDDPMConfig, EuclideanDDPMDiffuser
from mlkit.diffusion.euclidean_vpsde_diffuser import EuclideanVPSDEConfig, EuclideanVPSDEDiffuser
from mlkit.diffusion.time_scheduler import DiffusionTimeScheduler
from mlkit.util.mask.masker import Masker


class _DummyModel(nn.Module):
    def forward(self, **kwargs):  # noqa: ANN003
        x_t = kwargs["x_t"]
        return {"x": torch.zeros_like(x_t)}


def _mse_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return ((pred - target) ** 2 * mask).mean()


def _make_ddpm(n_steps: int = 100) -> EuclideanDDPMDiffuser:
    config = EuclideanDDPMConfig(n_discretization_steps=n_steps, ndim_micro_shape=1)
    scheduler = DiffusionTimeScheduler(num_train_timesteps=n_steps, idx_start=0)
    return EuclideanDDPMDiffuser(
        config=config,
        time_scheduler=scheduler,
        masker=Masker(ndim_mini_micro_shape=0),
        model=_DummyModel(),
        loss_fn=_mse_loss,
    )


def _make_vpsde(n_steps: int = 100) -> EuclideanVPSDEDiffuser:
    config = EuclideanVPSDEConfig(n_discretization_steps=n_steps, ndim_micro_shape=1)
    scheduler = DiffusionTimeScheduler(num_train_timesteps=n_steps, idx_start=0)
    return EuclideanVPSDEDiffuser(
        config=config,
        time_scheduler=scheduler,
        masker=Masker(ndim_mini_micro_shape=0),
        model=_DummyModel(),
        loss_fn=_mse_loss,
    )


def test_ddpm_accepts_euclidean_four_arg_signature() -> None:
    """Shared EuclideanDiffuser call sites must not raise TypeError on DDPM."""
    diffuser = _make_ddpm()
    batch = 4
    x_0 = torch.randn(batch, 2)
    mask = torch.ones_like(x_0)
    t_a = torch.full((batch,), diffuser.time_scheduler.get_timestep_index_start() - 1, dtype=torch.long)
    t_b = torch.full((batch,), 50, dtype=torch.long)

    result = diffuser.forward_process(x_0, t_a, t_b, mask, is_continuous_time=False)
    assert "x_t" in result
    assert result["x_t"].shape == x_0.shape


def test_ddpm_sentinel_matches_marginal_params() -> None:
    """Clean-sentinel call must reproduce the classic DDPM marginal a/b."""
    diffuser = _make_ddpm()
    batch = 8
    x_0 = torch.randn(batch, 2)
    mask = torch.ones_like(x_0)
    t_b = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.long)
    t_a = torch.full_like(t_b, diffuser.time_scheduler.get_timestep_index_start() - 1)

    result = diffuser.forward_process(x_0, t_a, t_b, mask)
    expected_a = diffuser.complete_micro_shape(diffuser.config.sqrt_alphas_cumprod[t_b])
    expected_b = diffuser.complete_micro_shape(diffuser.config.sqrt_1m_alphas_cumprod[t_b])
    assert torch.allclose(result["a"], expected_a)
    assert torch.allclose(result["b"], expected_b)


def test_ddpm_two_time_matches_ratio_formula() -> None:
    """Genuine two-time transition must match alpha_bar ratio closed form."""
    diffuser = _make_ddpm()
    batch = 4
    x_start = torch.randn(batch, 2)
    mask = torch.ones_like(x_start)
    t_a = torch.tensor([10, 20, 30, 40], dtype=torch.long)
    t_b = torch.tensor([50, 60, 70, 80], dtype=torch.long)

    result = diffuser.forward_process(x_start, t_a, t_b, mask)
    alpha_a = diffuser.config.alphas_cumprod[t_a]
    alpha_b = diffuser.config.alphas_cumprod[t_b]
    a_square = alpha_b / alpha_a
    expected_a = diffuser.complete_micro_shape(a_square.sqrt())
    expected_b = diffuser.complete_micro_shape((1 - a_square).clamp(min=0).sqrt())
    assert torch.allclose(result["a"], expected_a)
    assert torch.allclose(result["b"], expected_b)


def test_vpsde_accepts_euclidean_four_arg_signature() -> None:
    """Shared EuclideanDiffuser call sites must not raise / mis-bind on VPSDE."""
    diffuser = _make_vpsde()
    batch = 4
    x_0 = torch.randn(batch, 2)
    mask = torch.ones_like(x_0)
    t_a = torch.full((batch,), diffuser.time_scheduler.get_timestep_index_start() - 1, dtype=torch.long)
    t_b = torch.full((batch,), 50, dtype=torch.long)

    result = diffuser.forward_process(x_0, t_a, t_b, mask, is_continuous_time=False)
    assert "x_t" in result
    assert result["x_t"].shape == x_0.shape


def test_vpsde_sentinel_matches_sde_marginal() -> None:
    """Clean-sentinel call must reproduce VPSDE marginal a/b at t_b."""
    diffuser = _make_vpsde()
    batch = 8
    x_0 = torch.randn(batch, 2)
    mask = torch.ones_like(x_0)
    t_b = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.long)
    t_a = torch.full_like(t_b, diffuser.time_scheduler.get_timestep_index_start() - 1)

    result = diffuser.forward_process(x_0, t_a, t_b, mask)
    continuous_t_b = diffuser.time_scheduler.timestep_index_to_continuous_time(t_b)
    expected = diffuser.sde.forward_process(x_0, continuous_t_b, mask)
    assert torch.allclose(result["a"], expected["a"])
    assert torch.allclose(result["b"], expected["b"])
    assert torch.allclose(result["std"], expected["std"])


def test_vpsde_uses_t_b_not_t_a() -> None:
    """Regression: two-time call must use t_b for noise level (not silently bind t_a)."""
    diffuser = _make_vpsde()
    batch = 4
    x_0 = torch.randn(batch, 2)
    mask = torch.ones_like(x_0)
    t_a = torch.full((batch,), diffuser.time_scheduler.get_timestep_index_start() - 1, dtype=torch.long)
    t_b_small = torch.full((batch,), 5, dtype=torch.long)
    t_b_large = torch.full((batch,), 90, dtype=torch.long)

    result_small = diffuser.forward_process(x_0, t_a, t_b_small, mask)
    result_large = diffuser.forward_process(x_0, t_a, t_b_large, mask)

    # Larger t_b must produce larger noise scale b (std).
    assert (result_large["b"] > result_small["b"]).all()

    # If the bug were still present (discrete_t=t_a, mask=t_b), both would use
    # the same (near-clean) time and have nearly identical b.
    assert not torch.allclose(result_small["b"], result_large["b"])


def test_vpsde_two_time_matches_sde_transition() -> None:
    """Genuine intermediate→later call must match VPSDE.forward_from_t1_to_t2."""
    diffuser = _make_vpsde()
    batch = 4
    x_mid = torch.randn(batch, 2)
    mask = torch.ones_like(x_mid)
    t_a = torch.full((batch,), 30, dtype=torch.long)
    t_b = torch.full((batch,), 80, dtype=torch.long)
    continuous_t_a = diffuser.time_scheduler.timestep_index_to_continuous_time(t_a)
    continuous_t_b = diffuser.time_scheduler.timestep_index_to_continuous_time(t_b)

    torch.manual_seed(42)
    result = diffuser.forward_process(x_mid, t_a, t_b, mask)
    torch.manual_seed(42)
    expected = diffuser.sde.forward_from_t1_to_t2(x_mid, continuous_t_a, continuous_t_b)
    assert torch.allclose(result["x_t"], expected)
