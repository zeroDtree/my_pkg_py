from typing import Protocol, cast, runtime_checkable

import torch
from peft.tuners.lora.bnb import Linear4bit, Linear8bitLt
from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
from peft.tuners.lora.bnb import Linear8bitLt as LoraLinear8bitLt
from torch import Tensor
from torch.nn import Linear, Module


@runtime_checkable
class LinearLikeModule(Protocol):
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor | None

    def __call__(self, x: Tensor) -> Tensor: ...


def get_float_weight(model: LinearLikeModule) -> Tensor:
    """Get the float weight of a model.

    The model must expose `in_features`, `weight`, and optionally `bias`
    (e.g. `torch.nn.Linear`, `Linear8bitLt`, `Linear4bit`).

    Args:
        model: A linear-like module with `in_features` and `weight`.

    Returns:
        The reconstructed float weight tensor.
    """
    device = model.weight.device
    in_features = model.in_features
    with torch.no_grad():
        eye = torch.eye(in_features).to(device)
        w = model(eye)
        if model.bias is not None:
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = bool(model.weight.requires_grad)
    return w


def replace_module_with_linear(model: Module, target: type[Module]) -> None:
    """Replace all child modules of `target` type with plain `nn.Linear` layers.

    Args:
        model: The model whose children will be inspected.
        target: The module class to replace.
    """
    for name, module in model.named_children():
        if isinstance(module, target):
            linear_module = module
            if not isinstance(linear_module, LinearLikeModule):
                raise TypeError(f"Module {name} does not implement LinearLikeModule")
            in_features = linear_module.in_features
            out_features = linear_module.out_features
            bias = linear_module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(linear_module).data
                if bias and linear_module.bias is not None:
                    new_module.bias.data = linear_module.bias.data
            setattr(model, name, new_module)
        else:
            replace_module_with_linear(module, target)


def dequantize(model: Module, dtype: str) -> None:
    """Dequantize a model by replacing quantized linear layers with float ones.

    Args:
        model: The model to dequantize.
        dtype: Quantization dtype — `"int8"` or `"nf4"`.
    """
    target: type[Module] | None = None
    if dtype == "int8":
        target = LoraLinear8bitLt
    elif dtype == "nf4":
        target = LoraLinear4bit
    if target is not None:
        replace_module_with_linear(model=model, target=target)


class Config:
    in_features = 3
    out_features = 4
    device = "cuda"


def main() -> None:
    config = Config()
    print("get float weight")
    linear = Linear(config.in_features, config.out_features)
    print(linear.weight)
    w = get_float_weight(model=linear)
    print(w)

    print("int8 quant==============================================")
    base_linear_int8 = Linear(config.in_features, config.out_features)
    base_linear_int8.load_state_dict(linear.state_dict())
    m_int8 = Linear8bitLt(base_layer=base_linear_int8, adapter_name="default", has_fp16_weights=False)
    m_int8.to(config.device)
    print(m_int8.weight)
    w = get_float_weight(model=cast(LinearLikeModule, m_int8))
    print(linear.weight)
    print(w)

    print("nf4 quant==============================================")
    base_linear_nf4 = Linear(config.in_features, config.out_features)
    base_linear_nf4.load_state_dict(linear.state_dict())
    m_nf4 = Linear4bit(base_layer=base_linear_nf4, adapter_name="default")
    m_nf4.to(config.device)
    print(m_nf4.weight)
    print(linear.weight)
    w = get_float_weight(model=cast(LinearLikeModule, m_nf4))
    print(w)


if __name__ == "__main__":
    main()
