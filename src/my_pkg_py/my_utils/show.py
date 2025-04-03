import torch
from torch.utils.data import DataLoader


def show_info(
    model: torch.nn.Module = None,
    dataset: torch.utils.data.Dataset = None,
    optimizer: torch.optim.Optimizer = None,
    batch_size: int = 7,
):
    if model is not None:
        print("model info:")
        print(type(model))
        print(f"the total number of parameters = {sum(p.numel() for p in model.parameters())}")
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
