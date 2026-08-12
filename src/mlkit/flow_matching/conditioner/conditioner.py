from ...util.base_class.gm_conditioner import (
    Conditioner as _GMConditioner,
)
from ...util.base_class.gm_conditioner import (
    LossGuidanceConditioner as _LossGuidanceConditioner,
)
from ...util.decorators import inherit_docstrings


@inherit_docstrings
class Conditioner(_GMConditioner):
    r"""Base conditioner for flow matching guidance."""


@inherit_docstrings
class LGFMConditioner(_LossGuidanceConditioner, Conditioner):
    r"""Loss Guidance Flow Matching conditioner.

    Computes g(x_t) = -∇_{x_t} l(E[x_1|x_t], y) with RF posterior_mean_fn.
    """
