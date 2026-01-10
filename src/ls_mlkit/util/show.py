import torch
from tabulate import tabulate
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset


def table_print_dict(sample, prefix_priority_list=[], show_value=False):
    table_data = []
    # Collect and sort keys first
    keys = list(sample.keys())

    def key_priority(k):
        for i, prefix in enumerate(prefix_priority_list):
            if k.startswith(prefix):
                return (i,)
        return (len(prefix_priority_list),)

    keys.sort(key=key_priority)

    # Build table with sorted keys
    for k in keys:
        v = sample[k]
        shape_str = v.shape if isinstance(v, torch.Tensor) else ""
        length = len(v) if hasattr(v, "__len__") else ""
        value = ""
        if show_value:
            value = v
        table_data.append([k, type(v).__name__, shape_str, length, value])
    print(tabulate(table_data, headers=["Key", "Type", "Shape", "Length", "Value"], tablefmt="github"))


def show_info(
    model: Module | None = None,
    dataset: Dataset | None = None,
    optimizer: Optimizer | None = None,
    batch_size: int = 7,
) -> None:
    def print_kv_table(info: dict, title: str) -> None:
        table_data = []
        for k, v in info.items():
            if isinstance(v, float):
                v = f"{v:.6g}"
            table_data.append([k, v])
        print("\n" + title)
        print(tabulate(table_data, headers=["Key", "Value"], tablefmt="github"))

    def get_model_info(model: Module) -> dict:
        param_size = 4  # float32
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_bytes = total_params * param_size

        return {
            "class": model.__class__.__name__,
            "total_params": f"{total_params:,}",
            "trainable_params": f"{trainable_params:,}",
            "size_mb": total_bytes / (1024**2),
            "size_gb": total_bytes / (1024**3),
        }

    if model is not None:
        model_info = get_model_info(model)
        print_kv_table(model_info, title="Model Information")

    if dataset is not None:
        dataset_info = {
            "class": dataset.__class__.__name__,
            "batch_size": batch_size,
        }

        try:
            dataset_info["dataset_size"] = len(dataset)
        except TypeError:
            dataset_info["dataset_size"] = "unknown"

        print_kv_table(dataset_info, title="Dataset Information")

    if optimizer is not None:
        opt_info = {
            "class": optimizer.__class__.__name__,
            "num_param_groups": len(optimizer.param_groups),
        }

        print_kv_table(opt_info, title="Optimizer Information")

        for i, group in enumerate(optimizer.param_groups):
            group_info = {
                "lr": group.get("lr", "unknown"),
                "weight_decay": group.get("weight_decay", 0.0),
                "num_params": sum(p.numel() for p in group["params"]),
            }
            print_kv_table(group_info, title=f"Optimizer Param Group {i}")


def find_tensor_devices(obj, visited=None, path=""):
    if visited is None:
        visited = set()

    if id(obj) in visited:
        return {}
    visited.add(id(obj))

    devices = {}

    if isinstance(obj, torch.Tensor):
        devices[path] = obj.device
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            devices.update(find_tensor_devices(item, visited, f"{path}[{idx}]"))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            devices.update(find_tensor_devices(value, visited, f"{path}['{key}']"))
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            devices.update(find_tensor_devices(value, visited, f"{path}.{attr}" if path else attr))
    return devices
