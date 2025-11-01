from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from ls_mlkit.my_utils.utils_for_main import get_learing_rate_scheduler  # type: ignore
from ls_mlkit.my_utils.utils_for_main import get_new_save_dir  # type: ignore
from ls_mlkit.my_utils.utils_for_main import get_optimizer  # type: ignore
from ls_mlkit.my_utils.utils_for_main import get_run_name  # type: ignore
from ls_mlkit.my_utils.utils_for_main import get_train_class  # type: ignore
from ls_mlkit.my_utils.utils_for_main import load_checkpoint  # type: ignore


def get_dataset(cfg: DictConfig):
    from datasets import load_dataset

    image_size = cfg.dataset.image_size
    dataset_name = cfg.dataset.id

    # load full dataset
    train_dataset = load_dataset(dataset_name, split="train")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(train_dataset[:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.savefig("train_dataset.png")

    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    # apply same transform to three datasets
    train_dataset.set_transform(transform)

    return train_dataset, train_dataset, train_dataset


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):
    from diffusers import UNet2DModel

    from ls_mlkit.my_diffuser.conditioner import Conditioner
    from ls_mlkit.my_diffuser.euclidean_ddim_diffuser import EuclideanDDIMConfig, EuclideanDDIMDiffuser
    from ls_mlkit.my_diffuser.euclidean_ddpm_diffuser import EuclideanDDPMConfig, EuclideanDDPMDiffuser
    from ls_mlkit.my_diffuser.model_interface import Model4DiffuserInterface
    from ls_mlkit.my_diffuser.time_scheduler import TimeScheduler
    from ls_mlkit.my_utils.mask.image_masker import ImageMasker

    if model is None:
        model = UNet2DModel(
            sample_size=cfg.dataset.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    class MyModel(Model4DiffuserInterface, Module):
        def __init__(self, model: torch.nn.Module):
            Module.__init__(self)
            Model4DiffuserInterface.__init__(self)
            self.model: UNet2DModel = model

        def prepare_batch_data_for_input(self, batch: dict[str, Any]) -> dict[str, Any]:
            new_batch = {
                "x_0": batch["x_0"],
                "padding_mask": torch.ones_like(batch["x_0"], dtype=torch.bool),
            }
            return new_batch

        def get_model_device(self) -> torch.device:
            return next(self.model.parameters()).device

        def __call__(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
            p_noise: Tensor = self.model.forward(x_t, t, return_dict=False)[0]
            return {"x": p_noise}

    model = MyModel(model=model)
    time_scheduler = TimeScheduler(
        continuous_time_start=0.0,
        continuous_time_end=1.0,
        num_train_timesteps=cfg.diffuser.n_discretization_steps,
        num_inference_steps=cfg.diffuser.get("n_inference_steps", None),
    )

    def mse(predicted: Tensor, ground_truth: Tensor, mask: Tensor):
        from torch.nn.functional import mse_loss

        return mse_loss(predicted, ground_truth)

    conditioner_list = []

    DiffuserConfigClass = None
    DiffuserClass = None
    if cfg.diffuser.name == "DDPM":
        DiffuserConfigClass = EuclideanDDPMConfig
        DiffuserClass = EuclideanDDPMDiffuser
    elif cfg.diffuser.name == "DDIM":
        DiffuserConfigClass = EuclideanDDIMConfig
        DiffuserClass = EuclideanDDIMDiffuser
    else:
        raise ValueError(f"Invalid diffuser name: {cfg.diffuser.name}")
    diffusion_config = DiffuserConfigClass(
        n_discretization_steps=cfg.diffuser.n_discretization_steps,
        ndim_micro_shape=3,
        n_inference_steps=cfg.diffuser.get("n_inference_steps", None),
        eta=cfg.diffuser.get("eta", 0.0),
    )
    diffuser = DiffuserClass(
        config=diffusion_config,
        time_scheduler=time_scheduler,
        conditioner_list=cast(list[Conditioner], conditioner_list),
        loss_fn=mse,
        masker=ImageMasker(),
        model=model,
    )

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        diffuser = load_checkpoint(diffuser, final_model_ckpt_path)

    return diffuser


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        # examples is a list of dictionaries: [{"images": tensor1}, {"images": tensor2}, ...]
        # Extract the "images" from each example and stack them
        batch = []
        for example in examples:
            batch.append(example["images"])
        return {
            "x_0": torch.stack(batch),
            "mode": cfg.diffuser.mode,  # Use the mode from config
        }

    return collate_fn
