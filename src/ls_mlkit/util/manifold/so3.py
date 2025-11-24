r"""
SO(3): Special Orthogonal Group
"""

from typing import Callable, Tuple

import torch
from torch import Tensor

from .lie_group import LieGroup

EPS = 1e-8

from .so3_utils import exponential_map, logarithmic_map, trace, vector_to_skew_symmetric


class SO3(LieGroup):
    """SO(3): Special Orthogonal Group"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exp(self, p: Tensor = None, v: Tensor = None) -> Tensor:
        r"""Exponential map
        $$\exp_p(v)$$ map a point in tangent space $$T_p M$$ to a point on the manifold $$M$$
        $$\exp_p(v) = p \cdot \exp(p^{-1} v)$$
        if p is None, it will be set to the identity matrix
        """
        assert v is not None, "v is required"
        if p is None:
            return exponential_map(v)
        p_inv = self.inverse(p)
        v_e = p_inv @ v
        result = p @ exponential_map(v_e)
        return result

    def log(self, p: Tensor = None, q: Tensor = None) -> Tensor:
        r"""Logarithm map
        $$\log_p(q)$$ map a point on the manifold $$M$$ to a point in tangent space $$T_p M$$
        $$\log_p(q)=p\log(p^{-1} q)$$
        if p is None, it will be set to the identity matrix
        """
        assert q is not None, "q is required"
        if p is None:
            return logarithmic_map(q)
        p_inv = self.inverse(p)
        q_e = p_inv @ q
        result = p @ logarithmic_map(q_e)
        return result

    def random_tangent(self, p: Tensor, random_type: str = "gaussian", std: float = 1.0) -> Tensor:
        r"""Sample noise from $$T_p M$$"""
        noise = None
        if random_type == "gaussian":
            noise = torch.randn(p.shape[:-2] + (3,), dtype=p.dtype, device=p.device) * std
        else:
            raise ValueError(f"Invalid random type: {random_type}")
        assert noise is not None
        result = vector_to_skew_symmetric(noise)
        return result

    def metric(self, p: Tensor, v: Tensor, w: Tensor) -> Tensor:
        r"""Inner product
        $$<v, w>_p: \mathfrak{so}(3) \times \mathfrak{so}(3) \to \mathbb{R}$$ is the inner product at point $$p$$
        $$
        <v, w>_p = \frac{1}{2} \text{Tr}(v^T w)
        $$
        """
        result = 1 / 2 * trace(v.transpose(-1, -2) @ w)
        return result

    def grad(self, f: Callable, p: Tensor) -> Tensor:
        r"""Riemannian gradient of f at point p on SO(3)

        $$
        p \cdot skew(p^{-1} \nabla_p f(p))
        $$

        Args:
            f: Callable[[Tensor], Tensor], scalar function of p
            p: (..., 3, 3) point on SO(3)

        Returns:
            (..., 3, 3) gradient in the tangent space T_p SO(3)
        """
        p = p.clone().detach().requires_grad_(True)
        y = f(p)
        if y.ndim > 0:
            y = y.sum()

        # Euclidean gradient
        grad_euclid = torch.autograd.grad(y, p, create_graph=True)[0]  # (..., 3, 3)

        # project to tangent space T_p SO(3)
        grad_riemann = p @ (
            0.5 * (p.transpose(-1, -2) @ grad_euclid - (p.transpose(-1, -2) @ grad_euclid).transpose(-1, -2))
        )

        return grad_riemann

    def multiply(self, p, q):
        r"""Multiply in Group"""
        assert p.shape == q.shape, "p and q must have the same shape"
        result = p @ q
        return result

    def inverse(self, p):
        r"""Inverse in Group"""
        result = p.transpose(-1, -2)
        return result

    def identity(self, macro_shape: Tuple[int, ...] = tuple()):
        r"""Identity in Group"""
        result = torch.eye(3).view(*macro_shape, 3, 3)
        return result

    def left_translation(self, g, h):
        r"""
        $$L_g(h) = g \cdot h$$
        """
        result = g @ h
        return result


if __name__ == "__main__":
    pass
