import torch
from torch import Tensor


def rmsd_nodewise(predicted: Tensor, ground_truth: Tensor, mask: Tensor, eps: float = 1e-10):
    rmsds = (eps + torch.sum((predicted - ground_truth) ** 2, dim=-1)) ** 0.5
    return torch.sum(rmsds * mask) / torch.sum(mask)


def mse_nodewise(predicted: Tensor, ground_truth: Tensor, mask: Tensor):
    mses = torch.sum((predicted - ground_truth) ** 2, dim=-1)
    return torch.sum(mses * mask) / torch.sum(mask)


def rmsd(predicted: Tensor, ground_truth: Tensor, mask: Tensor, eps: float = 1e-10):
    assert predicted.shape == ground_truth.shape  # (bs, n, 3)
    assert predicted.shape[-1] == 3
    assert mask.shape == predicted.shape[:-1]  # (bs, n)

    sq_errors = (predicted - ground_truth) ** 2 * mask.unsqueeze(-1)
    mse = torch.sum(sq_errors) / (torch.sum(mask) * 3)
    rmsd = torch.sqrt(mse.clamp_min(eps))
    return rmsd


def mse(predicted: Tensor, ground_truth: Tensor, mask: Tensor):
    assert predicted.shape == ground_truth.shape
    assert predicted.shape[-1] == 3
    assert mask.shape == predicted.shape[:-1]
    sq_errors = (predicted - ground_truth) ** 2 * mask.unsqueeze(-1)
    mse = torch.sum(sq_errors) / (torch.sum(mask) * 3)
    return mse


def huber(predicted: Tensor, ground_truth: Tensor, mask: Tensor, delta: float = 1.0):
    assert predicted.shape == ground_truth.shape
    assert predicted.shape[-1] == 3
    assert mask.shape == predicted.shape[:-1]
    errors = (predicted - ground_truth) * mask.unsqueeze(-1)
    abs_errors = errors.abs()
    huber_loss = torch.where(abs_errors > delta, delta * (abs_errors - 0.5 * delta), 0.5 * errors**2)
    # huber_loss *= mask.unsqueeze(-1)
    return torch.sum(huber_loss) / (torch.sum(mask) * 3)


def huber_3_3(predicted: Tensor, ground_truth: Tensor, mask: Tensor, delta: float = 1.0):
    assert predicted.shape == ground_truth.shape
    assert predicted.shape[-1] == 3
    assert predicted.shape[-2] == 3
    assert mask.shape == predicted.shape[:-2]
    errors = (predicted - ground_truth) * mask.unsqueeze(-1).unsqueeze(-1)  # (..., n, 3, 3)
    abs_errors = errors.abs()
    huber_loss = torch.where(abs_errors > delta, delta * (abs_errors - 0.5 * delta), 0.5 * errors**2)
    # huber_loss *= mask.unsqueeze(-1)
    return torch.sum(huber_loss) / (torch.sum(mask) * 9)
