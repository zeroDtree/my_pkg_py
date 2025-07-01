import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from typing import Literal, List, Dict, Callable
import numpy as np
import wandb
from torch import Tensor
from torch.nn.parameter import Parameter


class Observer(object):
    function_mapping = {
        "weight_norm": lambda module: torch.sqrt(
            sum(
                torch.sum(p.data * p.data)
                for p in module.parameters()
                if p.requires_grad
            )
        ),
        "gradient_norm": lambda module: torch.sqrt(
            sum(
                torch.sum(p.grad.data * p.grad.data)
                for p in module.parameters()
                if p.grad is not None
            )
        ),
        "weights": lambda module: [
            p.detach().cpu() for p in module.parameters() if p.requires_grad
        ],
        "gradients": lambda module: [
            p.grad.detach().cpu() for p in module.parameters() if p.grad is not None
        ],
    }

    def __init__(
        self,
        model: Module = None,
        optimizer: Optimizer = None,
        scheduler: LambdaLR = None,
        dataset: Dataset | HFDataset = None,
        target_modules: List[str] = None,
        no_split_classes: List[str] = None,
    ):
        """
        target_modules (List[str], optional): The modules to observe. Defaults to None.
        if target_modules is not None, then no_split_classes and strategy is ignored.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.no_split_classes = no_split_classes
        self.target_modules = target_modules

    # get something=================================================================

    @torch.no_grad()
    def _get_something(
        model: Module,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
        function: Callable = None,
    ):
        info = dict()

        def __get_something(module: Module, prefix=""):
            if (
                len(list(module.named_children())) == 0
                or (
                    no_split_classes is not None
                    and module.__class__.__name__ in no_split_classes
                )
            ) and any(param.requires_grad for param in module.parameters()):
                info[prefix] = function(module)
                return
            for name, sub_module in module.named_children():
                sub_module_name = f"{prefix}.{name}" if prefix != "" else name
                __get_something(sub_module, sub_module_name)

        match strategy:
            case "all":
                something = function(model)
                return {"total_model": something}
            case "block":
                __get_something(model, "")
                return info
            case _:
                raise ValueError(f"Unsupported strategy: {strategy}")

    @torch.no_grad()
    def _get_target_modules(model: Module, target_modules: List[str]):
        info = dict()

        def __get_target_modules(module: Module, prefix=""):
            if any(target_module in prefix for target_module in target_modules):
                info[prefix] = module
                return
            for name, sub_module in module.named_children():
                sub_module_name = f"{prefix}.{name}" if prefix != "" else name
                __get_target_modules(sub_module, sub_module_name)

        __get_target_modules(model, "")
        return info

    @torch.no_grad()
    def _get_something_from_targets(
        model: Module = None,
        target_modules_dict: Dict[str, Module] = None,
        target_modules: List[str] = None,
        function: Callable = None,
    ):
        info = dict()
        if target_modules_dict is None:
            target_modules_dict = Observer._get_target_modules(model, target_modules)
        for module_path, module in target_modules_dict.items():
            info[module_path] = function(module)
        return info

    @torch.no_grad()
    def get_something_from_targets(self, function: Callable):
        return Observer._get_something_from_targets(
            model=self.model,
            target_modules_dict=None,
            target_modules=self.target_modules,
            function=function,
        )

    @torch.no_grad()
    def get_something(
        self,
        name,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        if self.target_modules is None:
            if no_split_classes is None:
                no_split_classes = self.no_split_classes
            return Observer._get_something(
                model=self.model,
                strategy=strategy,
                no_split_classes=no_split_classes,
                function=Observer.function_mapping[name],
            )
        return self.get_something_from_targets(function=Observer.function_mapping[name])

    @torch.no_grad()
    def get_weight_norm(
        self,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        return self.get_something("weight_norm", strategy, no_split_classes)

    @torch.no_grad()
    def get_gradient_norm(
        self,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        return self.get_something("gradient_norm", strategy, no_split_classes)

    @torch.no_grad()
    def get_weights(
        self,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        return self.get_something("weights", strategy, no_split_classes)

    @torch.no_grad()
    def get_gradients(
        self,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        return self.get_something("gradients", strategy, no_split_classes)

    @torch.no_grad()
    def _get_statistics(
        data: List[Tensor],
    ):
        flattened_tensor = torch.cat([item.reshape(-1) for item in data], dim=0)
        mean = flattened_tensor.mean()
        std = flattened_tensor.std()
        median = flattened_tensor.median()
        var = flattened_tensor.var()
        return {"mean": mean, "std": std, "median": median, "variance": var}

    @torch.no_grad()
    def get_statistics(
        self,
        name,
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
    ):
        something = self.get_something(
            name, strategy=strategy, no_split_classes=no_split_classes
        )
        return {
            key: Observer._get_statistics(value) for key, value in something.items()
        }

    # log something===============================================================

    @torch.no_grad()
    def log_statistics(
        self,
        strategy: Literal["all", "block", "both"] = "both",
        no_split_classes: List[str] = None,
        section="statistics/",
    ):
        if self.target_modules is not None:
            strategy = "block"
        result = {}
        if strategy in ["all", "both"]:
            weights_statistics_total_model = self.get_statistics(
                name="weights", strategy="all", no_split_classes=no_split_classes
            )
            gradients_statistics_total_model = self.get_statistics(
                name="gradients", strategy="all", no_split_classes=no_split_classes
            )
            weights_statistics = {
                section + "weight/" + module_path + "/" + k: v
                for module_path, info in weights_statistics_total_model.items()
                for k, v in info.items()
                if k in ["mean", "std", "median"]
            }
            gradients_statistics = {
                section + "gradient/" + module_path + "/" + k: v
                for module_path, info in gradients_statistics_total_model.items()
                for k, v in info.items()
                if k in ["mean", "std", "median"]
            }

            result.update(weights_statistics)
            result.update(gradients_statistics)
        if strategy in ["block", "both"]:
            weights_statistics_block = self.get_statistics(
                name="weights", strategy="block", no_split_classes=no_split_classes
            )
            gradients_statistics_block = self.get_statistics(
                name="gradients", strategy="block", no_split_classes=no_split_classes
            )
            weights_statistics_block = {
                section + "weight/" + module_path + "/" + k: v
                for module_path, info in weights_statistics_block.items()
                for k, v in info.items()
                if k in ["mean", "std", "median"]
            }
            gradients_statistics_block = {
                section + "gradient/" + module_path + "/" + k: v
                for module_path, info in gradients_statistics_block.items()
                for k, v in info.items()
                if k in ["mean", "std", "median"]
            }
            result.update(weights_statistics_block)
            result.update(gradients_statistics_block)
        wandb.log(result)

    @torch.no_grad()
    def log_histograms(
        data: Dict[str, List[Tensor]],
        bins: int = 16,
        section="histogram/",
        prefix="",
    ):
        results = dict()
        for key, data in data.items():
            flattened_tensor = torch.cat([item.reshape(-1) for item in data], dim=0)
            flattened_numpy = flattened_tensor.numpy()
            np_histogram = np.histogram(flattened_numpy, bins=bins)
            wandb_histogram = wandb.Histogram(np_histogram=np_histogram)
            results.update(
                {
                    section + prefix + key: wandb_histogram,
                }
            )
        wandb.log(results)

    @torch.no_grad()
    def log_distribution(
        self,
        name,
        bins: int = 16,
        section="observer/",
        strategy: Literal["all", "block"] = "all",
        no_split_classes: List[str] = None,
        desc="",
    ):
        something = self.get_something(
            name, strategy=strategy, no_split_classes=no_split_classes
        )
        Observer.log_histograms(
            data=something,
            bins=bins,
            section=section,
            prefix=name + "/" + desc,
        )

    @torch.no_grad()
    def log_weights_distribution(
        self,
        bins: int = 16,
        section="observer/",
        strategy: Literal["all", "block", "both"] = "both",
        no_split_classes: List[str] = None,
        desc="",
    ):
        if self.target_modules is not None:
            strategy = "block"
        if strategy in ["all", "both"]:
            self.log_distribution(
                name="weights",
                bins=bins,
                section=section,
                strategy="all",
                no_split_classes=no_split_classes,
                desc=desc,
            )
        if strategy in ["block", "both"]:
            self.log_distribution(
                name="weights",
                bins=bins,
                section=section,
                strategy="block",
                no_split_classes=no_split_classes,
                desc=desc,
            )

    @torch.no_grad()
    def log_gradients_distribution(
        self,
        bins: int = 16,
        section="observer/",
        strategy: Literal["all", "block", "both"] = "both",
        no_split_classes: List[str] = None,
        desc="",
    ):
        if self.target_modules is not None:
            strategy = "block"
        if strategy in ["all", "both"]:
            self.log_distribution(
                name="gradients",
                bins=bins,
                section=section,
                strategy="all",
                no_split_classes=no_split_classes,
                desc=desc,
            )
        if strategy in ["block", "both"]:
            self.log_distribution(
                name="gradients",
                bins=bins,
                section=section,
                strategy="block",
                no_split_classes=no_split_classes,
                desc=desc,
            )

    def log_gradient_norm(
        self,
        section="observer/",
        strategy: Literal["all", "block", "both"] = "both",
        no_split_classes: List[str] = None,
        desc="gradient_norm/",
    ):
        if self.target_modules is not None:
            strategy = "block"
        results = dict()
        if strategy in ["all", "both"]:
            gradient_norm_total_model = self.get_something(
                "gradient_norm", strategy="all", no_split_classes=no_split_classes
            )
            results.update(gradient_norm_total_model)
        if strategy in ["block", "both"]:
            gradient_norm_block = self.get_something(
                "gradient_norm", strategy="block", no_split_classes=no_split_classes
            )
            results.update(gradient_norm_block)
        sectioned_results = {
            section + desc + key: value for key, value in results.items()
        }
        wandb.log(sectioned_results)

    def log_weight_norm(
        self,
        section="observer/",
        strategy: Literal["all", "block", "both"] = "both",
        desc="",
        no_split_classes: List[str] = None,
    ):
        if self.target_modules is not None:
            strategy = "block"
        results = dict()
        if strategy in ["all", "both"]:
            weight_norm_total_model = self.get_something(
                "weight_norm", strategy="all", no_split_classes=no_split_classes
            )
            results.update(weight_norm_total_model)
        if strategy in ["block", "both"]:
            weight_norm_block = self.get_something(
                "weight_norm", strategy="block", no_split_classes=no_split_classes
            )
            results.update(weight_norm_block)
        sectioned_results = {
            section + "weight_norm/" + desc + key: value
            for key, value in results.items()
        }
        wandb.log(sectioned_results)
