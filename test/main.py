from ls_mlkit.my_diffuser.config import DiffusionConfig

from accelerate import Accelerator  # type: ignore
import torch

accelerator = Accelerator()

config = DiffusionConfig()


x = torch.Tensor([1, 2, 3])
x = x[:-0]
print(x)