"""official packages"""

from typing import Any, Callable, cast

import torch
from torch import Tensor

"""my packages"""
from ls_mlkit.my_utils.decorators import inherit_docstrings
from ls_mlkit.my_utils.nma.nma import get_nma_displacement_from_node_coordinates
from ls_mlkit.my_utils.vector_utils import get_vector_cosines

from .conditioner import LGDConditioner


def disp_loss_fn(
    disps: Tensor,
    gt_disps: Tensor,
    tgt_mask: Tensor,
    eps: float = 1e-8,
    lambda_angle: float = 1.0,
    lambda_ampl: float = 1.0,
) -> Tensor:
    r"""
    Compute the combined dynamic loss:

    .. math::

        l_{\text{dynamic}} = \lambda_{\text{angle}} \cdot l_{\text{angle}} + \lambda_{\text{amplitude}} \cdot l_{\text{amplitude}}

        l_{\text{angle}} = \sum_{1 \leq i < j \leq n} \left\| \cos^2(v(\hat{x}_0^i), v(\hat{x}_t^j)) - \cos^2(v(x_0^i), v(x_t^j)) \right\|^2

        l_{\text{amplitude}} = \sum_{1 \leq i \leq n} \left\| \log \left( \frac{\|v(\hat{x}_0^i)\|}{\sum_{1 \leq j \leq n} \|v(\hat{x}_0^j)\|} \right) - \log \left( \frac{\|v(x_0^i)\|}{\sum_{1 \leq j \leq n} \|v(x_0^j)\|} \right) \right\|^2

    Args:
        disps: shape = (..., n_nodes, 3)
        gt_disps: shape = (..., n_nodes, 3)
        mask: shape = (..., n_nodes), 1 for node whose loss should be computed
        eps: small value to avoid log(0)
        lambda_angle: weight for angle loss
        lambda_ampl: weight for amplitude loss

    Returns:
        loss: scalar tensor
    """
    n = disps.shape[-2]
    device = disps.device
    # --- angle loss ---
    cosines = get_vector_cosines(disps)  # (..., n_nodes, n_nodes)
    gt_cosines = get_vector_cosines(gt_disps)  # (..., n_nodes, n_nodes)
    # create pairwise mask
    matrix_mask = torch.einsum("...i,...j->...ij", tgt_mask, tgt_mask)  # (..., n_nodes, n_nodes)
    # avoid double-counting diagonal
    matrix_mask = matrix_mask * (1.0 - torch.eye(n, device=device)).view(*([1] * (disps.dim() - 2)), n, n)
    n_pairs = torch.sum(matrix_mask)
    l_angle = torch.sum((cosines**2 - gt_cosines**2) ** 2 * matrix_mask) / (n_pairs + eps)

    # --- amplitude loss ---
    # compute norm for each node
    disp_norm = torch.norm(disps, dim=-1)  # (..., n_nodes)
    gt_disp_norm = torch.norm(gt_disps, dim=-1)  # (..., n_nodes)
    # apply mask
    disp_norm = disp_norm * tgt_mask
    gt_disp_norm = gt_disp_norm * tgt_mask
    # compute normalized proportions
    disp_sum = torch.sum(disp_norm, dim=-1, keepdim=True) + eps  # (..., 1)
    gt_disp_sum = torch.sum(gt_disp_norm, dim=-1, keepdim=True) + eps  # (..., 1)
    p_hat = disp_norm / disp_sum  # (..., n_nodes)
    p_gt = gt_disp_norm / gt_disp_sum  # (..., n_nodes)
    # log-relative squared difference
    l_amplitude = torch.sum((torch.log(p_hat + eps) - torch.log(p_gt + eps)) ** 2, dim=-1)
    # average over batch
    if l_amplitude.dim() > 0:
        l_amplitude = torch.mean(l_amplitude)

    # --- total loss ---
    loss = lambda_angle * l_angle + lambda_ampl * l_amplitude
    return loss


@inherit_docstrings
class NMAConditioner(LGDConditioner):

    def __init__(self, guidance_scale: float = 0.2):
        """Initialize the NMAConditioner

        Args:
            guidance_scale (float, optional): the guidance scale of the conditioner. Defaults to 0.2.
        """
        super().__init__(guidance_scale)
        self.tgt_mask: Tensor = None
        self.gt_disps: Tensor = None
        self.posterior_mean_fn: Callable = None

    def prepare_condition_dict(self, train: bool = True, *args: list[Any], **kwargs: dict[Any, Any]) -> dict[str, Any]:
        """
        Get something that is needed to compute the conditional loss and that not in (x, t, padding_mask, posterior_mean_fn)

        Required: tgt_mask, gt_disps(or x_0 and padding_mask)
        """
        if train:
            tgt_mask = kwargs.get("tgt_mask", None)
            assert tgt_mask is not None, "tgt_mask is required"
            posterior_mean_fn = kwargs.get("posterior_mean_fn", None)
            assert posterior_mean_fn is not None, "posterior_mean_fn is required"

            gt_disps = kwargs.get("gt_disps", None)
            if gt_disps is None:
                x_0 = kwargs.get("x_0", None)
                padding_mask = kwargs.get("padding_mask", None)
                assert x_0 is not None, "x_0 is required if gt_disps is not provided"
                assert padding_mask is not None, "padding_mask is required if gt_disps is not provided"
                gt_disps = get_nma_displacement_from_node_coordinates(
                    cast(Tensor, x_0),
                    node_mask=cast(Tensor, padding_mask),
                )  # (..., n_nodes, 3)
            condition_dict = {
                "tgt_mask": tgt_mask,
                "gt_disps": gt_disps,
                "posterior_mean_fn": posterior_mean_fn,
            }
            return condition_dict
        else:
            return dict()

    def set_condition(self, *args: list[Any], **kwargs: dict[Any, Any]):
        self.tgt_mask = cast(Tensor, kwargs.get("tgt_mask", None))
        self.gt_disps = cast(Tensor, kwargs.get("gt_disps", None))
        self.posterior_mean_fn = kwargs.get("posterior_mean_fn", None)
        assert self.tgt_mask is not None, "tgt_mask is required"
        assert self.gt_disps is not None, "gt_disps is required"
        assert self.posterior_mean_fn is not None, "posterior_mean_fn is required"
        self.ready = True

    @torch.enable_grad()
    def compute_conditional_loss(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        r"""
        Compute the conditional loss

        "..." can be arbitrary number of dimensions, for example, batch size.
        Args:
            x.shape = (..., n_nodes, 3)
            t.shape = (..., )
            padding_mask.shape = (..., n_nodes) 1 means non-padding region

        Condition:
            tgt_mask.shape = (..., n_nodes) 1 means target region
            gt_disps.shape = (..., n_nodes, 3)
            posterior_mean_fn: function to compute posterior mean, input: x_t, t, padding_mask, output: posterior mean
        """

        tgt_mask = self.tgt_mask
        gt_disps = self.gt_disps
        posterior_mean_fn = self.posterior_mean_fn
        assert torch.all(torch.logical_or(tgt_mask == 0, padding_mask == 1)), "where mask is 1, padding_mask must be 1"
        cache: dict[str, Tensor] = dict()
        cache[r"$$E[x_0|x_t]$$"] = posterior_mean_fn(x_t, t, padding_mask)
        disps = get_nma_displacement_from_node_coordinates(
            cache[r"$$E[x_0|x_t]$$"], node_mask=padding_mask
        )  # (..., n_nodes, 3)
        loss = disp_loss_fn(disps=disps, gt_disps=gt_disps, tgt_mask=tgt_mask)
        return loss
