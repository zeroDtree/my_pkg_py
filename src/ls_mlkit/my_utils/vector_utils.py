from typing import Callable

import torch
from torch import Tensor


def get_vector_cosines(vectors: Tensor) -> Tensor:
    """
    Args:
        vectors: torch.Tensor, shape (..., n_nodes, 3)
    Returns:
        cosines: torch.Tensor, shape (..., n_nodes * n_nodes)
    """
    rows_norms = torch.norm(vectors, dim=-1)
    rows_norms = torch.where(rows_norms == 0, torch.tensor([1], device=rows_norms.device), rows_norms)
    normalised_vectors = torch.einsum("...nd,...n->...nd", vectors, 1 / rows_norms)

    cosines = torch.einsum("...ij,...kj->...ik", normalised_vectors, normalised_vectors)

    return cosines


def get_cosines_and_amplitudes(vectors: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    cosines = get_vector_cosines(vectors)
    torch.linalg.norm: Callable
    amplitudes = torch.linalg.norm(vectors, dim=-1)
    amplitudes = amplitudes * mask  # type: ignore
    amplitudes = torch.nn.functional.normalize(amplitudes, p=2, dim=-1)  # type: ignore #
    return cosines, amplitudes
