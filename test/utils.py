from omegaconf import DictConfig
from torch import Tensor
from typing import Any, cast
import matplotlib.pyplot as plt
import torch
import os
import math
from torch.nn import Module


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

    from ls_mlkit.my_diffuser.euclidean_ddpm_diffuser import EuclideanDDPMConfig, EuclideanDDPMDiffuser
    from ls_mlkit.my_diffuser.euclidean_ddim_diffuser import EuclideanDDIMConfig, EuclideanDDIMDiffuser
    from ls_mlkit.my_diffuser.time_scheduler import TimeScheduler
    from ls_mlkit.my_utils.mask.image_masker import ImageMasker
    from ls_mlkit.my_diffuser.conditioner import Conditioner
    from ls_mlkit.my_diffuser.model_interface import Model4DiffuserInterface

    if model is None:
        model = UNet2DModel(
            sample_size=cfg.dataset.image_size,  # the target image resolution
            in_channels=7,  # the number of input channels, 3 for RGB images + 3 for masked images + 1 for mask
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
                "masked_x_0": batch["masked_x_0"],
                "inpainting_mask": batch["inpainting_mask"],
            }
            return new_batch

        def get_model_device(self) -> torch.device:
            return next(self.model.parameters()).device

        def __call__(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> dict:
            masked_x_0 = kwargs.get("masked_x_0", None)
            assert masked_x_0 is not None, "masked_x_0 is required"
            inpainting_mask = kwargs.get("inpainting_mask", None)
            assert inpainting_mask is not None, "inpainting_mask is required"
            x_t = torch.cat((x_t, masked_x_0, inpainting_mask), dim=-3)
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

    if final_model_ckpt_path is not None:
        # Handle different checkpoint formats
        if final_model_ckpt_path.endswith(".safetensors"):
            print(f"Loading safetensors checkpoint: {final_model_ckpt_path}")
            try:
                from safetensors.torch import load_file

                checkpoint = load_file(final_model_ckpt_path)
                # For safetensors, the state_dict is directly the checkpoint
                diffuser.load_state_dict(checkpoint, strict=False)
                print(f"Loaded diffuser from safetensors: {final_model_ckpt_path}")
            except ImportError:
                print("safetensors library not found. Install with: pip install safetensors")
                raise
            except Exception as e:
                print(f"Error loading safetensors file: {e}")
                # Fallback: try to load the underlying UNet model instead
                print("Trying to load UNet model from the checkpoint...")
                diffuser.model.model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded UNet model from safetensors: {final_model_ckpt_path}")
        else:
            print(f"Loading PyTorch checkpoint: {final_model_ckpt_path}")
            checkpoint = torch.load(final_model_ckpt_path, map_location="cpu", weights_only=False)
            # print(f"Loaded PyTorch checkpoint with keys: {list(checkpoint.keys())}")
            if "model" in checkpoint:
                # If saved via pipeline, the diffuser is under 'model' key
                diffuser.load_state_dict(checkpoint["model"])
            else:
                # If saved directly as state_dict
                diffuser.load_state_dict(checkpoint)
            print(f"Loaded diffuser from PyTorch checkpoint: {final_model_ckpt_path}")

    return diffuser


def get_run_name(cfg: DictConfig, suffix: str = "", prefix: str = ""):
    run_name = f"{cfg.dataset.name}-{cfg.model.name.replace('/','_').replace('-','_')}-lr:{cfg.optimizer.lr}"
    if cfg.train.train_strategy in ["steps"]:
        run_name += f"-n_steps:{cfg.train.n_steps}"
    if cfg.train.train_strategy in ["epochs"]:
        run_name += f"-n_epochs:{cfg.train.n_epochs}"
    run_name = f"{prefix}{run_name}{suffix}"
    return run_name


def get_learing_rate_scheduler(optimizer, accelerator, train_set, cfg: DictConfig):
    from ls_mlkit.my_scheduler.lr_scheduler_factory import get_lr_scheduler

    if cfg.train.train_strategy in ["epochs"]:
        effective_batch_size = accelerator.num_processes * cfg.train.batch_size
        n_training_steps = math.ceil(1.0 * len(train_set) * cfg.train.n_epochs / effective_batch_size)
    elif cfg.train.train_strategy in ["steps"]:
        n_training_steps = cfg.train.n_steps
    else:
        raise ValueError(f"Train Strategy {cfg.train.train_strategy} is not supported")
    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        n_warmup_steps=cfg.train.n_warmup_steps,
        n_training_steps=n_training_steps,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
    )
    return lr_scheduler


def get_new_save_dir(save_dir, cfg: DictConfig, suffix: str = "", prefix: str = ""):
    if save_dir is None:
        save_dir = "checkpoints"
    model_name = str(cfg.model.name.replace("/", "_").replace("-", "_"))
    dataset_name = str(cfg.dataset.id)
    lr = str(cfg.optimizer.lr)
    batch_size = str(cfg.train.batch_size)
    optimizer_name = str(cfg.optimizer.name)
    diffusion_mode = str(cfg.diffuser.mode)
    new_save_dir = os.path.join(
        save_dir, f"{model_name}/{dataset_name}/-lr:{lr}-bs:{batch_size}-{optimizer_name}-{diffusion_mode}"
    )
    new_save_dir = f"{prefix}{new_save_dir}{suffix}"
    return new_save_dir


def get_optimizer(model, cfg: DictConfig):
    import torch  # type: ignore

    optimizer_config = dict(cfg.optimizer)
    optimizer_class_name = optimizer_config.pop("name")
    optimizer = eval("torch.optim." + optimizer_class_name)(model.parameters(), **optimizer_config)
    return optimizer


def get_train_class():
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ls_mlkit.my_pipeline import MyDistributedPipeline, MyTrainingConfig

    return MyDistributedPipeline, MyTrainingConfig


def get_collate_fn(cfg: DictConfig):
    
    def collate_fn(examples):
        # examples is a list of dictionaries: [{"images": tensor1}, {"images": tensor2}, ...]
        # Extract the "images" from each example and stack them
        batch = []
        for example in examples:
            batch.append(example["images"])
        
        # Stack all images in the batch
        x_0 = torch.stack(batch)  # Shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x_0.shape
        
        # Generate inpainting masks - keep right half, mask left half
        inpainting_masks = []
        masked_images = []
        
        for i in range(batch_size):
            # Create mask tensor (1 = keep, 0 = mask out)
            # Keep right half, mask left half
            mask = torch.ones(1, height, width)
            mask[:, :, :width // 2] = 0  # Mask left half
            
            inpainting_masks.append(mask)
            
            # Create masked image (set masked regions to 0)
            masked_image = x_0[i].clone()
            masked_image = masked_image * mask  # Zero out the left half
            
            masked_images.append(masked_image)
        
        # Stack masks and masked images
        inpainting_mask = torch.stack(inpainting_masks)  # Shape: [batch_size, 1, height, width]
        masked_x_0 = torch.stack(masked_images)  # Shape: [batch_size, channels, height, width]
            
        result = {
            "x_0": x_0,
            "inpainting_mask": inpainting_mask,  # Note: keeping the typo to match the model expectation
            "masked_x_0": masked_x_0,
            "mode": cfg.diffuser.mode,  # Use the mode from config
        }
        print(f"result.keys() = {result.keys()}")
        return result
    return collate_fn
