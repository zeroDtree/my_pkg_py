from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module
from transformers.trainer import Accelerator

from ls_mlkit.util.utils_for_main import get_learing_rate_scheduler  # type: ignore
from ls_mlkit.util.utils_for_main import get_new_save_dir  # type: ignore
from ls_mlkit.util.utils_for_main import get_optimizer  # type: ignore
from ls_mlkit.util.utils_for_main import get_run_name  # type: ignore
from ls_mlkit.util.utils_for_main import get_train_class  # type: ignore
from ls_mlkit.util.utils_for_main import load_checkpoint  # type: ignore


def get_dataset(cfg: DictConfig):
    from sklearn.datasets import make_moons
    from torch.utils.data import Dataset

    class MyDataset(Dataset):
        def __init__(self):
            super().__init__()
            self.data = Tensor(make_moons(256, noise=0.15)[0])

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    train_dataset = MyDataset()

    return train_dataset, train_dataset, train_dataset


def get_collate_fn(cfg: DictConfig):
    from sklearn.datasets import make_moons

    def collate_fn(examples):
        # examples is a list of dictionaries: [{"images": tensor1}, {"images": tensor2}, ...]
        # Extract the "images" from each example and stack them
        batch = examples
        x_1 = torch.stack(batch)
        return {"gt_data": Tensor(make_moons(256, noise=0.15)[0]), "padding_mask": torch.ones_like(x_1)}

    return collate_fn


def get_model(cfg: DictConfig, model=None, final_model_ckpt_path=None):

    from torch import nn

    from ls_mlkit.flow_matching.euclidean_ot_fm import EuclideanOTFlow, EuclideanOTFlowConfig
    from ls_mlkit.flow_matching.time_scheduler import FlowMatchingTimeScheduler
    from ls_mlkit.model.model_for_pipeline import ModelForPipeline
    from ls_mlkit.util.mask.image_masker import ImageMasker

    class BaseModel(Module):
        def __init__(self, dim: int = 2, h: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, dim)
            )

        def forward(self, x_t: Tensor, t: Tensor, *args, **kwarg) -> Tensor:
            return self.net(torch.cat((t, x_t), -1))

    base_model = BaseModel()

    class MyModel(Module):
        def __init__(self, model: Module):
            super().__init__()
            self.model = model

        def forward(self, **batch: dict[str, Any]) -> Tensor:
            x_t: Tensor = batch["x_t"]
            t: Tensor = batch["t"]
            t = t.unsqueeze(-1)
            return {"x": self.model(x_t, t, return_dict=False)}

    def mse(predicted: Tensor, ground_truth: Tensor, mask: Tensor):
        from torch.nn.functional import mse_loss

        return mse_loss(predicted, ground_truth)

    model = MyModel(model=base_model)

    model_for_pipeline = ModelForPipeline(model=model)

    time_scheduler = FlowMatchingTimeScheduler(
        num_train_timesteps=cfg.flow.n_discretization_steps,
        num_inference_steps=cfg.flow.n_inference_steps,
    )

    flow_config = EuclideanOTFlowConfig(
        n_discretization_steps=cfg.flow.n_discretization_steps,
        ndim_micro_shape=1,
        n_inference_steps=cfg.flow.n_inference_steps,
    )
    flow = EuclideanOTFlow(
        config=flow_config,
        time_scheduler=time_scheduler,
        model=model_for_pipeline,
        masker=ImageMasker(ndim_mini_micro_shape=0),
        loss_fn=mse,
    )

    if final_model_ckpt_path is not None and final_model_ckpt_path != "":
        flow = load_checkpoint(flow, final_model_ckpt_path)

    import torch.nn.functional as F
    from sklearn.datasets import make_moons
    from torch.optim import AdamW

    class MoonsClassifier(Module):
        def __init__(self, dim: int = 2, h=64, n_labels=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, h),
                nn.ELU(),
                nn.Linear(h, h),
                nn.ELU(),
                nn.Linear(h, h),
                nn.ELU(),
                nn.Linear(h, n_labels),
            )

        def forward(self, x):
            logits = self.net(x)
            return logits

    classifier_model = MoonsClassifier()
    optimizer = AdamW(classifier_model.parameters())

    for i in range(2000):
        x, c = make_moons(256, noise=0.15)
        x = Tensor(x)
        c = Tensor(c).long()
        logits = classifier_model(x)
        loss = F.cross_entropy(logits, c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    x, c = make_moons(100, noise=0.15)
    x = Tensor(x)
    c = Tensor(c).long()
    logits = classifier_model(x)
    p_l = torch.argmax(logits, dim=-1)
    print(p_l)
    print(c)
    acc = (p_l == c).float().sum() / 100
    print(f"acc={acc}")

    classifier_model = classifier_model.to(Accelerator().device)

    from ls_mlkit.diffusion.conditioner.conditioner import LGDConditioner

    class ClassifierConditioner(LGDConditioner):
        def __init__(self, classifier_model, guidance_scale: float = 1.0):
            super().__init__(guidance_scale)
            self.classifier_model = classifier_model

        def prepare_condition_dict(self, train=True, *args, **kwargs):
            """
            Get something that is needed to compute the conditional loss and that not in (x, t, padding_mask, posterior_mean_fn)

            Required: tgt_mask, label (or gt_data and padding_mask)
            """
            tgt_mask = kwargs.get("tgt_mask", None)
            assert tgt_mask is not None, "tgt_mask is required"
            posterior_mean_fn = kwargs.get("posterior_mean_fn", None)
            assert posterior_mean_fn is not None, "posterior_mean_fn is required"
            if train:
                gt_data = kwargs.get("gt_data")
                logits = classifier_model(gt_data)
                p_l = torch.argmax(logits, dim=-1)
                return {
                    "tgt_mask": tgt_mask,
                    "label": p_l,
                    "posterior_mean_fn": posterior_mean_fn,
                }
            else:
                label = kwargs.get("sampling_condition", None)
                assert label is not None, "label is required"
                return {
                    "tgt_mask": tgt_mask,
                    "label": label,
                    "posterior_mean_fn": posterior_mean_fn,
                }

        def set_condition(self, *args, **kwargs):
            self.tgt_mask = kwargs.get("tgt_mask", None)
            self.label = kwargs.get("label")
            self.posterior_mean_fn = kwargs.get("posterior_mean_fn")
            self.ready = True

        def compute_conditional_loss(self, p_gt_data, padding_mask):
            c = self.label
            c = c.squeeze(-1).long()
            logits = self.classifier_model(p_gt_data)
            loss = F.cross_entropy(logits, c)
            return loss

    classifier_conditioner = ClassifierConditioner(classifier_model=classifier_model, guidance_scale=7.0)
    samling_hook = flow.get_condition_pre_update_in_step_fn_hook([classifier_conditioner])
    sampling_hook_handlers = flow.register_hooks([samling_hook])
    train_hook = flow.get_condition_post_compute_loss_hook([classifier_conditioner])
    train_hook_handlers = flow.register_hooks([train_hook])
    return {"model": flow, "train_hook_handlers": train_hook_handlers, "sampling_hook_handlers": sampling_hook_handlers}
