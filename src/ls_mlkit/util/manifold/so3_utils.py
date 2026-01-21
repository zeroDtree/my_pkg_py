r"""
SO3 Utils
"""

from typing import Tuple

import torch
from torch import Tensor

from ...util.decorators import cache_to_disk
from ...util.interp import interp

EPS = 1e-8


def get_macro_shape(x: Tensor, ndim_micro_shape: int) -> Tuple[int, ...]:
    return x.shape[:-ndim_micro_shape]


def flatten_batch_dimension(x: Tensor, ndim_micro_shape: int) -> Tensor:
    return x.view(-1, *x.shape[-ndim_micro_shape:])


def trace(A: Tensor) -> Tensor:
    """
    Args:
        A.shape: (..., 3, 3)
    Returns:
        shape: (..., )
    """
    return torch.diagonal(A, dim1=-1, dim2=-2).sum(dim=-1)  # (..., )


def vector_to_skew_symmetric(v: Tensor) -> Tensor:
    r"""
    Hat map from vector space $$\mathbb{R}^3$$ to Lie algebra $$\mathfrak{so}(3)$$
    $$
    (x,y,z) \to \begin{pmatrix}
    0 & -z & y \\
    z & 0 & -x \\
    -y & x & 0
    \end{pmatrix}
    $$
    Args:
        v.shape: (..., 3)
    Returns: 
        shape: (..., 3, 3)
    """
    macro_shape = get_macro_shape(x=v, ndim_micro_shape=1)
    hat_v = torch.zeros([*macro_shape, 3, 3], dtype=v.dtype, device=v.device)
    hat_v[..., 0, 1], hat_v[..., 0, 2], hat_v[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return hat_v + -hat_v.transpose(-1, -2)


def skew_symmetric_to_vector(hat_v: Tensor) -> Tensor:
    r"""
    Map from skew-symmetric matrix to vector
    $$\mathfrak{so}(3) \mapsto \mathbb{R}^3$$
    $$
    \begin{pmatrix}
    0 & -z & y \\
    z & 0 & -x \\
    -y & x & 0
    \end{pmatrix} \to  (x,y,z) 
    $$
    Args:
        hat_v.shape: (..., 3, 3)
    Returns:
        shape: (..., 3)
    """
    macro_shape = get_macro_shape(x=hat_v, ndim_micro_shape=2)
    v = torch.zeros([*macro_shape, 3], dtype=hat_v.dtype, device=hat_v.device)
    v[..., 0], v[..., 1], v[..., 2] = -hat_v[..., 1, 2], hat_v[..., 0, 2], -hat_v[..., 0, 1]
    return v


def skew_symmetric_to_angle(A: Tensor) -> Tensor:
    r""" 
    $$
    \begin{pmatrix}
    0 & -z & y \\
    z & 0 & -x \\
    -y & x & 0
    \end{pmatrix} \to  \sqrt{x^2 + y^2 + z^2}
    $$
    Args:
        A.shape: (..., 3, 3)
    Returns:
        shape: (..., )
    """
    return torch.sqrt(A[..., 0, 1] ** 2 + A[..., 0, 2] ** 2 + A[..., 1, 2] ** 2)


def unit_skew_symmetric(A: Tensor) -> Tensor:
    r"""
    get the unit skew-symmetric matrix
    Args:
        A.shape: (..., 3, 3)
    Returns:
        shape: (..., 3, 3)
    """
    theta = skew_symmetric_to_angle(A)
    return A / (theta[..., None, None] + EPS)


def rotation_matrix_to_angle(R: Tensor) -> Tensor:
    r"""
    $$
    \theta = \arccos(\frac{Tr(R)-1}{2})
    $$
    Args:
        R.shape: (..., 3, 3)
    Returns:
        shape: (..., )
    """
    theta = torch.arccos(((trace(R) - 1) / 2).clamp(min=-1 + EPS, max=1 - EPS))  # (..., )
    return theta


def logarithmic_map(R: Tensor) -> Tensor:
    r"""
    Logarithmic map from SO(3) to so(3), this is the matrix logarithm
    $$SO(3) \mapsto \mathfrak{so}(3)$$
    $$
    \begin{align*}
    \theta &= \arccos(\frac{Tr(R)-1}{2})\\
    \log(R) &= \frac{\theta}{2\sin(\theta)} (R - R^T)
    \end{align*}
    $$
    Args:    
        R.shape: (..., 3, 3)
    Returns:
        shape: (..., 3, 3)
    """
    theta = rotation_matrix_to_angle(R)  # (..., )
    log_R = (theta / (2 * torch.sin(theta) + EPS))[..., None, None] * (R - R.transpose(-1, -2))  # (..., 3, 3)
    return log_R


def exponential_map(A: Tensor) -> Tensor:
    r"""
    Exponential map from vector space of $$\mathfrak{so}(3)$$ to SO(3), this is the matrix
    $$\mathfrak{so}(3) \mapsto SO(3)$$
    $$
    \begin{align*}
    \theta &= \sqrt{A_{0,1}^2 + A_{0,2}^2 + A_{1,2}^2}\\
    B &= A / \theta\\
    \exp(A) &= I + \sin(\theta) B + (1 - \cos(\theta)) B^2
    \end{align*}
    $$
    Args:
        A.shape: (..., 3, 3)
    Returns:
        shape: (..., 3, 3)
    """
    theta = skew_symmetric_to_angle(A)
    unit_A = unit_skew_symmetric(A)
    macro_shape = get_macro_shape(x=A, ndim_micro_shape=2)
    ndim_macro_shape = len(macro_shape)
    exp_A = (
        torch.eye(3).view(*[1 for _ in range(ndim_macro_shape)], 3, 3).to(dtype=A.dtype, device=A.device)  # (..., 3, 3)
        + torch.sin(theta)[..., None, None] * unit_A  # (..., 3, 3)
        + (1 - torch.cos(theta))[..., None, None] * unit_A @ unit_A  # (..., 3, 3)
    )
    return exp_A


L_default = 2000


def f_igso3(omega: Tensor, c: Tensor, L: int = L_default) -> Tensor:
    r"""Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here,
    $$\sigma = \sqrt{2} * \epsilon$$, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2 when defined for the canonical inner product on SO3,
    $$<u, v>_{SO3} = Tr(u v^T)/2$$

    Args:
        omega: (...,), i.e. the angle of rotation associated with rotation matrix
        c: (1,), variance parameter of IGSO(3), maps onto time in Brownian motion
        L: (1,)Truncation level
    Returns:

    $$
    \sum_{i=0}^{L-1} (2l+1) e^{-l(l+1)c/2} \sin (\omega (l +1/2)) / sin(\omega/2)
    $$
    """
    ls = torch.arange(L)  # of shape [L,]
    s = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * c / 2).view(*([1] * omega.ndim), L)  # (..., L)

    numerator = torch.sin(
        omega[..., None] * (ls + 1 / 2).view(*([1] * omega.ndim), L)
    )  # (..., 1) * (..., L) = (..., L)
    denominator = torch.sin(omega / 2).unsqueeze(-1)  # (..., 1)

    # Add small epsilon to prevent division by zero when omega is close to 0 or 2π
    denominator_safe = torch.clamp(torch.abs(denominator), min=1e-8) * torch.sign(denominator)
    denominator_safe = torch.where(torch.abs(denominator) < 1e-8, torch.ones_like(denominator) * 1e-8, denominator)

    result = s * numerator / denominator_safe  # (..., L)
    result = result.sum(dim=-1)

    # Ensure result is always positive and finite
    result = torch.clamp(result, min=1e-12)
    return result


def d_logf_d_omega(omega: Tensor, c: Tensor, L: int = L_default) -> Tensor:
    r"""
    Score function of IGSO(3) distribution.

    $$
    \frac{d}{d\omega} \log f(\omega, c, L)
    $$
    Args:
        omega: (...,), i.e. the angle of rotation associated with rotation matrix
        c: (1,), variance parameter of IGSO(3), maps onto time in Brownian motion
        L: (1,)Truncation level
    Returns:
        (..., )
    """
    omega = omega.clone().detach().requires_grad_(True)
    f_val = f_igso3(omega, c, L)

    # Clamp f_val to avoid log(0) or log(negative)
    # Use a small positive value to prevent numerical instability
    f_val_clamped = torch.clamp(f_val, min=1e-8)

    log_f = torch.log(f_val_clamped)
    return torch.autograd.grad(log_f.sum(), omega)[0]


def igso3_density(Rt: Tensor, c: Tensor, L: int = L_default) -> Tensor:
    r"""
    IGSO3 density with respect to the volume form on SO(3)
    Args:
        Rt: (..., 3, 3), rotation matrix
        c: (1,), variance parameter of IGSO(3), maps onto time in Brownian motion
        L: (1,)Truncation level
    Returns:
        (..., )
    """
    omega = rotation_matrix_to_angle(Rt)
    return f_igso3(omega, c, L)


def igso3_density_angle(omega: Tensor, c: Tensor, L: int = L_default) -> Tensor:
    r"""
    $$((1-\cos(\omega)) / \pi ) f$$
    Args:
        omega: (...,), i.e. the angle of rotation associated with rotation matrix
        c: (1,), variance parameter of IGSO(3), maps onto time in Brownian motion
        L: (1,)Truncation level
    Returns:
        (..., )
    """
    return f_igso3(omega, c, L) * (1 - torch.cos(omega)) / torch.pi


def igso3_score(R: Tensor, c: Tensor, L: int = L_default) -> Tensor:
    r"""
    grad_R log IGSO3(R; I_3, c)
    $$
    \nabla_R \log IG_{SO3}(R; I_3, c) = R \frac{log(R)}{\omega(R)} \frac{d}{d\omega} \log f(\omega, c, L)
    $$

    Args:
        R: (..., 3, 3), rotation matrix
        c: (1,), variance parameter of IGSO(3), maps onto time in Brownian motion
        L: (1,)Truncation level
    Returns:
        (..., 3, 3)
    """
    omega = rotation_matrix_to_angle(R)  # (..., )
    unit = torch.einsum("...ij,...jk->...ik", R, logarithmic_map(R)) / omega[:, None, None]  # (..., 3, 3)
    return unit * d_logf_d_omega(omega, c, L)[..., None, None]


@cache_to_disk(root_datadir="cache")
def calculate_igso3(
    *, num_sigma: int, num_omega: int, min_sigma: float, max_sigma: float, discrete_omega=None, discrete_sigma=None
) -> dict[str, Tensor]:
    r"""calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_sigma: number of different sigmas for which to compute igso3
            quantities.
        num_omega: number of point in the discretization in the angle of
            rotation.
        min_sigma, max_sigma: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    if discrete_omega is None:
        discrete_omega = torch.linspace(0, torch.pi, num_omega + 1)[1:]  # [num_omega, ]
    else:
        discrete_omega = discrete_omega

    # Exponential noise schedule.  This choice is closely tied to the
    # scalings used when simulating the reverse time SDE. For each step n,
    # discrete_sigma[n] = min_eps^(1-n/num_eps) * max_eps^(n/num_eps)

    if discrete_sigma is None:
        discrete_sigma = (
            10 ** torch.linspace(torch.log10(min_sigma), torch.log10(max_sigma), num_sigma + 1)[1:]
        )  # [num_sigma, ]
    else:
        discrete_sigma = discrete_sigma

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    # $$\pi/\omega$$ is the length of the interval in the angle of rotation
    pdf_vals = torch.stack(
        [igso3_density_angle(discrete_omega, sigma**2) for sigma in discrete_sigma]
    )  # [num_sigma, num_omega]
    cdf_vals = torch.stack([pdf.cumsum(dim=-1) / num_omega * torch.pi for pdf in pdf_vals])  # [num_sigma, num_omega]

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    score_norm = torch.stack(
        [d_logf_d_omega(discrete_omega, sigma**2) for sigma in discrete_sigma]
    )  # [num_sigma, num_omega]

    # Compute the standard deviation of the score norm for each sigma
    r"""
    $$
    \sqrt{\mathbb{E}_{\omega} || \frac{d}{d\omega} f(\omega, c, L)||_2^2}
    $$
    """
    exp_score_norms = torch.sqrt(
        torch.sum(score_norm**2 * pdf_vals, axis=1) / torch.sum(pdf_vals, axis=1)
    )  # [num_sigma, ]
    return {
        "cdf": cdf_vals,  # [num_sigma, num_omega]
        "score_norm": score_norm,  # [num_sigma, num_omega]
        "exp_score_norms": exp_score_norms,  # [num_sigma, ]
        "discrete_omega": discrete_omega,  # [num_omega, ]
        "discrete_sigma": discrete_sigma,  # [num_sigma, ]
    }


def inverse_transform_sampling(shape: Tuple[int, ...], cdf: Tensor, discrete_omega: Tensor) -> Tensor:
    r"""
    Sample uses the inverse cdf to sample an angle of rotation from
    ``IGSO(3)``

    Args:
        shape: shape of the sampled angles of rotation.
        cdf: (num_omega,), cdf of the IGSO(3) distribution
        discrete_omega: (num_omega, ), discrete angles of rotation
    Returns:
        sampled angles of rotation. ``(*shape,)``
    """

    result = interp(
        x=torch.rand(shape).to(cdf.device),  # uniform distribution over [0, 1]
        xp=discrete_omega,
        fp=cdf,
    )  # (*shape,)
    return result
