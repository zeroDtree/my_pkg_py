from typing import Callable, Optional, Tuple, Union
from my_pipeline.pipeline import BasePipeline, LogConfig, TrainingConfig
import torch
from tqdm import tqdm
import datasets


class ModelConfig:
    def __init__(
        self,
        n_steps: int,
        image_width: int,
        image_height: int,
        image_channels: int,
    ):
        self.n_steps = n_steps
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels


class SamplingConfig:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples


class DDPMPipeline(BasePipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR],
        training_config: TrainingConfig,
        log_config: LogConfig,
        model_config: ModelConfig,
        sampling_config: SamplingConfig,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            optimizers=optimizers,
            training_config=training_config,
            log_config=log_config,
            collate_fn=collate_fn,
        )
        self.model_config = model_config
        self.sampling_config = sampling_config

    def compute_loss(self, model, batch):
        return model(**batch)

    @torch.no_grad()
    def generate(self) -> torch.Tensor:
        model = self.model
        n_samples = self.sampling_config.n_samples
        n_steps = self.model_config.n_steps
        image_width = self.model_config.image_width
        image_height = self.model_config.image_height
        image_channels = self.model_config.image_channels
        device = self.training_config.device
        model.eval()

        # $$x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$$
        x = torch.randn([n_samples, image_channels, image_width, image_height], device=device)

        # Remove noise for $T$ steps
        for t_ in tqdm(range(n_steps)):
            # $t$
            t = n_steps - t_ - 1
            # Sample from $$\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$$
            x = model.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
        return x


if __name__ == "__main__":
    pipeline = DDPMPipeline(
        model=None,
        dataset=None,
        optimizers=None,
        training_config=None,
        log_config=None,
        model_config=None,
        sampling_config=None,
        collate_fn=None,
    )
