import abc
from typing import Any

import torch
from torch import Tensor


class Model4DiffuserInterface(abc.ABC):
    def __init__(
        self,
    ):
        pass

    @abc.abstractmethod
    def prepare_batch_data_for_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare batch data for input

        Args:
            batch (dict[str, Any]): the batch of data

        Returns:
            dict[str, Any]: the prepared batch of data
        """

    @abc.abstractmethod
    def get_model_device(self) -> torch.device:
        """Get the device of the model

        Returns:
            torch.device: the device of the model
        """

    @abc.abstractmethod
    def __call__(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
        r"""Call the model

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask
            *args: additional arguments
            **kwargs: additional keyword arguments

        Returns:
            dict: the output of the model
        """
