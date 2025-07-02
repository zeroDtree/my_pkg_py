"""Official package"""

import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple, Any
from tqdm.auto import tqdm
import abc

"""custom package"""
from .config import DiffusionConfig
from ls_mlkit.my_utils import MaskerInterface, BioCAOnlyMasker
from .loss_utils import rmsd
from .conditioner import Conditioner
from .model_interface import ModelInterface4Diffuser


class Diffuser(Module, abc.ABC):
    r"""
    `compute_loss`,`q_xt_x_0`,`compute_loss` must be overridden.
    """

    def __init__(
        self,
        model: ModelInterface4Diffuser,
        config: DiffusionConfig,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = rmsd,
        masker: MaskerInterface = BioCAOnlyMasker(),
        conditioner_list: list[Conditioner] = [],
    ):
        super().__init__()  # type: ignore
        self.model: ModelInterface4Diffuser = model
        self.config: DiffusionConfig = config
        self.loss_fn = loss_fn
        self.masker = masker
        self.conditioner_list = conditioner_list

    @abc.abstractmethod
    def q_xt_x_0(self, x_0: Tensor, t: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward process
        $$q(x_t|x_0)$$
        return (expectation, standard_deviation)
        """

    @abc.abstractmethod
    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> Tensor: ...

    @abc.abstractmethod
    def sample_xtm1_conditional_on_xt(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor: ...

    def sample_a_time_step(self, macro_shape: Tuple[int, ...]) -> Tensor:
        """
        Sample a timestep
        """
        return torch.randint(0, self.config.n_discretization_steps, macro_shape)

    def get_accumulated_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        accumulated_conditional_score = torch.zeros_like(x_t)
        for conditioner in self.conditioner_list:
            accumulated_conditional_score += conditioner.get_conditional_score(x_t, t, padding_mask)
        return accumulated_conditional_score

    def forward(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]):
        return self.compute_loss(batch, *args, **kwargs)

    def get_macro_and_micro_shape(self, x: Tensor):
        ndim_micro_shape = self.config.ndim_micro_shape
        return x.shape[:-ndim_micro_shape], x.shape[-ndim_micro_shape:]

    def get_macro_shape(self, x: Tensor):
        return x.shape[: -self.config.ndim_micro_shape]

    def get_something_proper_shape(self, base: Tensor, something: Tensor) -> Tensor:
        macro_shape = self.get_macro_shape(base)
        assert something.shape == macro_shape, "something.shape must be equal to x.macro_shape"
        something = something.view(*macro_shape, *[1 for _ in range(self.config.ndim_micro_shape)])
        return something.to(base.device)

    @torch.no_grad()  # type: ignore
    def sample_x0_unconditionally(self, shape: Tuple[int, ...]):
        r"""
        $$
        P_0 \leftarrow  P_1 \leftarrow \cdots \leftarrow P_T
        $$
        """
        macro_shape = shape[: -self.config.ndim_micro_shape]
        config = self.config
        masker = self.masker
        model = self.model
        device = model.get_model_device()
        x_t = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(config.n_discretization_steps))):
            t = i
            t = torch.ones(macro_shape, device=device) * t
            no_padding_mask = masker.get_full_bright_mask(x_t)
            x_t = self.sample_xtm1_conditional_on_xt(x_t, t, no_padding_mask)

        return x_t

    @torch.no_grad()  # type: ignore
    def inpainting_x0_unconditionally(self, x_0: Tensor, padding_mask: Tensor, inpainting_mask: Tensor):
        r"""
        $$
        P_0 \leftarrow  P_1 \leftarrow \cdots \leftarrow P_T
        $$
        """
        shape = x_0.shape
        config = self.config
        macro_shape = shape[: -config.ndim_micro_shape]
        masker = self.masker
        model: ModelInterface4Diffuser = self.model
        device = model.get_model_device()

        x_t = torch.randn(shape, device=device)
        x_0 = masker.apply_mask(x_0, padding_mask)
        x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)
        for i in tqdm(reversed(range(config.n_discretization_steps))):
            t = i
            t = torch.ones(macro_shape, device=device) * t
            x_t = self.sample_xtm1_conditional_on_xt(x_t, t, padding_mask)
            x_t = masker.apply_inpainting_mask(x_0, x_t, inpainting_mask)
            x_t = masker.apply_mask(x_t, padding_mask)

        return x_t
