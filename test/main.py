import torch
import abc


class A(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abc.abstractmethod
    def f(self, x: torch.Tensor) -> torch.Tensor:
        pass


class B(A):
    def f(self, x: torch.Tensor) -> torch.Tensor:
        return x


b = B()

b.f(torch.Tensor([1, 2, 3]))
