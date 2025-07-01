import torch
from torch.utils.data import DataLoader
from tabulate import tabulate


def table_print_dict(sample, prefix_priority_list=["pocket_", "ligand_"]):
    table_data = []
    # Collect and sort keys first
    keys = list(sample.keys())

    def key_priority(k):
        for i, prefix in enumerate(prefix_priority_list):
            if k.startswith(prefix):
                return (i, k.split("_")[-1])
        return (len(prefix_priority_list), k)

    keys.sort(key=key_priority)

    # Build table with sorted keys
    for k in keys:
        v = sample[k]
        shape_str = v.shape if isinstance(v, torch.Tensor) else ""
        length = len(v) if hasattr(v, "__len__") else ""
        table_data.append([k, type(v).__name__, shape_str, length])
    print(tabulate(table_data, headers=["Key", "Type", "Shape", "Length"], tablefmt="github"))


def show_info(
    model: torch.nn.Module = None,
    dataset: torch.utils.data.Dataset = None,
    optimizer: torch.optim.Optimizer = None,
    batch_size: int = 7,
):
    def get_model_size(model):
        # 每个参数占用 4 bytes (32-bit float)
        param_size = 4
        total_params = sum(p.numel() for p in model.parameters())
        total_size_bytes = total_params * param_size

        # 转换为 MB 和 GB
        total_size_kb = total_size_bytes / 1024
        total_size_mb = total_size_bytes / (1024 * 1024)  # bytes to MB
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)  # MB to GB

        return {
            "total_params": total_params,
            "total_size_kb": total_size_kb,
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_gb,
        }

    if model is not None:
        print("model info:")
        print(type(model))
        print(
            f"the total number of parameters = {sum(p.numel() for p in model.parameters())},{get_model_size(model)['total_size_gb']} GB"
        )
    if dataset is not None:
        print("dataset info:")
        print(type(dataset))
        eval_set_loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)
        for X, y in eval_set_loader:
            print(f"Shape of X: {X.shape} {X.dtype}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        del eval_set_loader
    if optimizer is not None:
        print("optimizer info:")
        print(type(optimizer))
        print(optimizer)


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
