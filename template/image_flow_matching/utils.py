from typing import Any

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

    from ls_mlkit.flow_matching.fm import EuclideanFlow, EuclideanFlowConfig

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

    class MyModel(Module):
        def __init__(self, model: Module):
            super().__init__()
            self.model = model

        def forward(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            return self.model(x_t, t, return_dict=False)[0]

    def mse(predicted: Tensor, ground_truth: Tensor, mask: Tensor):
        from torch.nn.functional import mse_loss

        return mse_loss(predicted, ground_truth)

    model4fm = MyModel(model=model)

    flow_config = EuclideanFlowConfig(
        n_discretization_steps=cfg.flow.n_discretization_steps,
        ndim_micro_shape=3,
        n_inference_steps=cfg.flow.n_inference_steps,
    )
    flow = EuclideanFlow(
        config=flow_config,
        model=model4fm,
        loss_fn=mse,
    )

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        flow = load_checkpoint(flow, final_model_ckpt_path)

    return flow


def get_collate_fn(cfg: DictConfig):
    def collate_fn(examples):
        # examples is a list of dictionaries: [{"images": tensor1}, {"images": tensor2}, ...]
        # Extract the "images" from each example and stack them
        batch = []
        for example in examples:
            batch.append(example["images"])
        return {
            "x_1": torch.stack(batch),
        }

    return collate_fn
