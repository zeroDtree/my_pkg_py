import torch
from torch import Tensor


def interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    1D linear interpolation in PyTorch, similar to numpy.interp.
    Supports x with arbitrary shape.

    Args:
        x  (Tensor): The x-coordinates to interpolate at. Shape (...,)
        xp (Tensor): The x-coordinates of the data points. Shape (M,)
        fp (Tensor): The y-coordinates of the data points. Shape (M,)

    Returns:
        Tensor: Interpolated values at x. Shape (...,)
    """
    # Flatten x for searchsorted
    x_flat = x.reshape(-1)

    # Find indices of bins
    indices = torch.searchsorted(xp, x_flat, right=False)

    # Clamp to [1, len(xp)-1]
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Gather points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    f0 = fp[indices - 1]
    f1 = fp[indices]

    # Linear interpolation with numerical stability
    denominator = x1 - x0
    # Add small epsilon to prevent division by zero
    denominator_safe = torch.where(torch.abs(denominator) < 1e-8, torch.ones_like(denominator) * 1e-8, denominator)
    slope = (f1 - f0) / denominator_safe
    y_flat = f0 + slope * (x_flat - x0)

    # Reshape back to x's shape
    return y_flat.view_as(x)


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    r"""
    1D linear interpolation in PyTorch, similar to numpy.interp.
    Supports x with arbitrary shape.
    ... := macro_shape

    Args:
        x  (Tensor): The x-coordinates to interpolate at. Shape (...,)
        xp (Tensor): The x-coordinates of the data points. Shape (M,)
        fp (Tensor): The y-coordinates of the data points. Shape (..., M,)

    Returns:
        Tensor: Interpolated values at x. Shape (...,)
    """
    macro_shape = x.shape
    M = xp.shape[-1]

    # flatten x
    x_flat = x.reshape(-1)
    x_flat.shape[0]

    # find indices of bins
    indices = torch.searchsorted(xp, x_flat, right=False)
    indices = torch.clamp(indices, 1, M - 1)

    # construct indices for gather
    # indices need to be expanded to (..., M) dimension
    indices0 = indices - 1  # (B, )
    indices1 = indices  # (B, )

    # expand fp to (batch, M)
    fp_flat = fp.reshape(-1, M)

    # gather needs indices to be (batch, 1)
    f0 = torch.gather(fp_flat, 1, indices0.unsqueeze(1))  # (B, 1)
    f1 = torch.gather(fp_flat, 1, indices1.unsqueeze(1))  # (B, 1)

    # interpolation with numerical stability
    x0 = xp[indices0]  # (B,)
    x1 = xp[indices1]  # (B,)
    denominator = (x1 - x0).unsqueeze(1)  # (B, 1)
    # Add small epsilon to prevent division by zero
    denominator_safe = torch.where(torch.abs(denominator) < 1e-8, torch.ones_like(denominator) * 1e-8, denominator)
    slope = (f1 - f0) / denominator_safe  # (B, 1)
    y_flat = f0 + slope * (x_flat - x0).unsqueeze(1)  # (B, 1)

    return y_flat.view(*macro_shape)


if __name__ == "__main__":
    x = torch.tensor([0, 1, 3, 4, 5])
    xp = torch.tensor([0, 2, 3, 4, 6])
    fp = torch.tensor([10, 20, 30, 40, 60])
    print(interp(x, xp, fp))
