"""Shared dataset utilities for examples."""

from typing import Any, cast

import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset


def get_hf_image_dataset(cfg: DictConfig):
    """Load a HuggingFace image dataset, apply standard preprocessing, and return (train, val, test) splits.

    All three splits point to the same dataset object (no explicit val/test split).
    """
    from datasets import load_dataset
    from torchvision import transforms

    image_size = cfg.dataset.image_size
    dataset_name = cfg.dataset.id

    train_dataset = load_dataset(dataset_name, split="train")

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    preview_dataset = cast(Any, train_dataset)
    for i, image in enumerate(preview_dataset.select(range(4))["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    fig.savefig("train_dataset.png")

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

    cast(Any, train_dataset).set_transform(transform)

    return train_dataset, train_dataset, train_dataset


class MoonsDataset(Dataset):
    """Sklearn moons dataset wrapped as a PyTorch Dataset."""

    def __init__(self, n_samples: int = 1024, noise: float = 0.15):
        super().__init__()
        from sklearn.datasets import make_moons

        self.data = Tensor(make_moons(n_samples, noise=noise)[0])

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
