from my_pipeline.pipeline import LogConfig, TrainingConfig
from my_pipeline.ddpm_pipeline import DDPMPipeline, ModelConfig, SamplingConfig
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path=None,config_name=None, version_base=None)
def main(cfg:DictConfig):
    print(OmegaConf.to_yaml(cfg))
    from my_model.diffusion.ddpm import DDPM
    from my_model.diffusion.unet import UNet
    from my_dataset.minist_cifar import get_dataset
    from my_utils import seed_everything
    from torchvision.transforms import ToPILImage, Resize
    import os

    seed_everything(2025)

    training_config = TrainingConfig(
        n_epochs=cfg.n_epochs, batch_size=32, device="cuda", save_dir="checkpoints", save_strategy="steps", save_steps=500
    )
    log_config = LogConfig(log_dir="logs", log_steps=5, log_strategy="steps")
    model_config = ModelConfig(n_steps=1000, image_width=64, image_height=64, image_channels=1)
    sampling_config = SamplingConfig(n_samples=64)

    dataset, _, _ = get_dataset("mnist", size=(model_config.image_width, model_config.image_height))
    """"
    dataset[0][0].shape=(1, 64, 64)"
    """

    def collate_fn(batch):
        x0 = torch.concat([item[0].unsqueeze(0) for item in batch], dim=0)
        return {
            "x0": x0,
        }

    epsilon_model = UNet(
        image_channels=model_config.image_channels,
        n_channels=64,
        ch_mults=(1, 2, 2, 4),
        is_attn=(False, False, True, True),
        n_blocks=2,
    )
    ddpm_model = DDPM(
        eps_model=epsilon_model,
        n_steps=model_config.n_steps,
        device=training_config.device,
    )
    optimizers = (torch.optim.Adam(ddpm_model.parameters(), lr=1e-4, weight_decay=0.0), None)

    pipeline = DDPMPipeline(
        model=ddpm_model,
        dataset=dataset,
        optimizers=optimizers,
        training_config=training_config,
        log_config=log_config,
        model_config=model_config,
        sampling_config=sampling_config,
        collate_fn=collate_fn,
    )

    pipeline.train()
    samples = pipeline.generate()
    """
    samples.shape=(batch_size, c, h, w)
    """
    os.makedirs("generated_images", exist_ok=True)
    # Save each sample as an image
    to_pil = ToPILImage()
    to_original_size = Resize((28, 28))  # Resize back to original MNIST size
    for i, sample in enumerate(samples):
        img = to_pil(sample)
        img = to_original_size(img)
        img.save(f"generated_images/sample_{i}.png")


if __name__ == "__main__":
    main()