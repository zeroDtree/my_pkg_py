import torch
from torch import Tensor


def rmsd(predicted: Tensor, ground_truth: Tensor, mask: Tensor, eps: float = 1e-10):
    rmsds = (eps + torch.sum((predicted - ground_truth) ** 2, dim=-1)) ** 0.5
    return torch.sum(rmsds * mask) / torch.sum(mask)
