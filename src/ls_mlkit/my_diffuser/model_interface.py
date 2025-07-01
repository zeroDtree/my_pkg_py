import abc
import torch
from torch import Tensor
from typing import Any


class ModelInterface4Diffuser(abc.ABC):
    def __init__(
        self,
    ):
        pass

    @abc.abstractmethod
    def prepare_batch_data_for_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def get_model_device(self) -> torch.device:
        pass

    @abc.abstractmethod
    def __call__(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        pass
