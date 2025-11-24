from typing import Callable, cast

import torch
from torch import Tensor

from ls_mlkit.diffusion.conditioner.utils import get_accumulated_conditional_score


def get_posterior_mean_fn(config, score: Tensor = None, score_fn: Callable = None):
    r"""Get the posterior mean function

    Args:
        score (Tensor, optional): the score of the sample
        score_fn (Callable, optional): the function to compute score

    Returns:
        Callable: the posterior mean function
    """
    from ls_mlkit.diffusion.euclidean_ddpm_diffuser import EuclideanDDPMConfig

    def _ddpm_posterior_mean_fn(
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
    ):
        r"""
        Args:
            x_t: shape=(..., n_nodes, 3)
            t: shape=(...), dtype=torch.long

        For the case of DDPM sampling, the posterior mean is given by

        .. math::

            E[x_0|x_t] = \frac{1}{\sqrt{\bar{\alpha}(t)}}(x_t + (1 - \bar{\alpha}(t))\nabla_{x_t}\log p_t(x_t))

        """
        nonlocal config, score, score_fn
        assert score is not None or score_fn is not None, "either score or score_fn must be provided"
        if score is None:
            score = score_fn(x_t, t, padding_mask)
        config = cast(EuclideanDDPMConfig, config.to(t))
        alpha_bar_t = config.alphas_cumprod[t]  # macro_shape
        alpha_bar_t.view(*alpha_bar_t.shape, *([1] * config.ndim_micro_shape))
        x_0 = (x_t + (1 - alpha_bar_t) * score) / torch.sqrt(alpha_bar_t)
        return x_0

    return _ddpm_posterior_mean_fn


def get_ddpm_hook(conditioner_list):

    def _hook_fn(**kwargs):
        nonlocal conditioner_list

        loss = kwargs.get("loss")
        x_0 = kwargs.get("clean_data")
        x_t = kwargs.get("x_t")
        t = kwargs.get("t", None)
        noise = kwargs.get("noise", None)
        predicted_noise = kwargs.get("predicted_noise")
        padding_mask = kwargs.get("padding_mask")
        a = kwargs.get("a")
        b = kwargs.get("b")
        loss_fn = kwargs.get("loss_fn")
        mode = kwargs.get("mode")
        config = kwargs.get("config")

        p_uc_score = -predicted_noise / b
        gt_uc_score = -noise / b

        tgt_mask = padding_mask
        for conditioner in conditioner_list:
            if not conditioner.is_enabled():
                continue
            conditioner.set_condition(
                **{
                    **conditioner.prepare_condition_dict(
                        train=True,
                        **{
                            "tgt_mask": tgt_mask,
                            "x_0": x_0,
                            "padding_mask": padding_mask,
                            "posterior_mean_fn": get_posterior_mean_fn(config=config, score=p_uc_score, score_fn=None),
                        },
                    ),
                }
            )

        acc_c_score = get_accumulated_conditional_score(conditioner_list, x_t, t, padding_mask)
        gt_score = gt_uc_score + acc_c_score

        # Scale and compute conditioned loss
        p_uc_score = b * p_uc_score
        gt_score = b * gt_score
        total_loss = loss_fn(p_uc_score, gt_score, padding_mask)
        kwargs["loss"] = total_loss
        return kwargs

    return _hook_fn
