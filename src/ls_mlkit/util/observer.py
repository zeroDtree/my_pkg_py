from typing import Callable, Dict, List, Literal

import torch
from datasets import Dataset as HFDataset
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset


def weight_norm_fn(module: Module):
    """Compute the weight norm of a module

    Args:
        module (Module): the module to compute the weight norm

    Returns:
        float: the weight norm of the module
    """
    return torch.sqrt(sum(torch.sum(p.data * p.data) for p in module.parameters() if p.requires_grad))


def gradient_norm_fn(module: Module):
    """Compute the gradient norm of a module

    Args:
        module (Module): the module to compute the gradient norm

    Returns:
        float: the gradient norm of the module
    """
    return torch.sqrt(sum(torch.sum(p.grad.data * p.grad.data) for p in module.parameters() if p.grad is not None))


def weights_fn(module: Module) -> list[Tensor]:
    """Get the weights of a module

    Args:
        module (Module): the module to get the weights

    Returns:
        list: the weights of the module
    """
    return [p.detach().cpu() for p in module.parameters() if p.requires_grad]


def gradients_fn(module: Module) -> list[Tensor]:
    """Get the gradients of a module

    Args:
        module (Module): the module to get the gradients

    Returns:
        list: the gradients of the module
    """
    return [p.grad.detach().cpu() for p in module.parameters() if p.grad is not None]


class Observer(object):
    function_mapping = {
        "weight_norm": weight_norm_fn,
        "gradient_norm": gradient_norm_fn,
        "weights": weights_fn,
        "gradients": gradients_fn,
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
        """Initialize the Observer


        Args:
            model (Module, optional): the model to observe. Defaults to None.
            optimizer (Optimizer, optional): the optimizer to observe. Defaults to None.
            scheduler (LambdaLR, optional): the scheduler to observe. Defaults to None.
            dataset (Dataset | HFDataset, optional): the dataset to observe. Defaults to None.
            target_modules (List[str], optional): the modules to observe. Defaults to None. if target_modules is not None, then no_split_classes and strategy is ignored.
            no_split_classes (List[str], optional): the classes to not split. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.no_split_classes = no_split_classes
        self.target_modules = target_modules

    # get something=================================================================

    @torch.no_grad()
    @staticmethod
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
                or (no_split_classes is not None and module.__class__.__name__ in no_split_classes)
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
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def _get_statistics(data: List[Tensor]):
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
        something = self.get_something(name, strategy=strategy, no_split_classes=no_split_classes)
        return {key: Observer._get_statistics(value) for key, value in something.items()}
