from ls_mlkit.my_diffuser import Diffuser, ModelInterface4Diffuser, DiffusionConfig  # type: ignore
from ls_mlkit.my_utils import HF_MIRROR  # type: ignore
from typing import Any
import torch  # type: ignore
from torch import Tensor
from ls_mlkit.my_utils import ImageMasker # type: ignore

# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline  # type: ignore

from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput  # type: ignore

model_id = "google/ddpm-celebahq-256"


class Unet(ModelInterface4Diffuser):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model: torch.nn.Module = model

    def get_model_device(self):
        return next(self.model.parameters()).device

    def prepare_batch_data_for_input(self, batch: dict[str, Any]):
        return batch

    def __call__(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        result: UNet2DOutput = self.model.forward(x_t, t)  # type: ignore
        return result["sample"]


# load model and scheduler
ddpm = PNDMPipeline.from_pretrained(model_id).to("cuda")  # type: ignore #

model = Unet(ddpm.unet)  # type: ignore

diffuser = Diffuser(model=model,diffusion_config= , masker=ImageMasker())