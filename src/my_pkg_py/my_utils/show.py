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
        print(
            f"the total number of parameters = {sum(p.numel() for p in model.parameters())}"
        )
    if dataset is not None:
        print("dataset info:")
        print(type(dataset))
        eval_set_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, drop_last=True
        )
        for X, y in eval_set_loader:
            print(f"Shape of X: {X.shape} {X.dtype}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break
        del eval_set_loader
    if optimizer is not None:
        print("optimizer info:")
        print(type(optimizer))
        print(optimizer)
