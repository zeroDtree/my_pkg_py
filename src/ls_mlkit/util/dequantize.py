import torch
from peft.tuners.lora.bnb import Linear4bit, Linear8bitLt
from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
from peft.tuners.lora.bnb import Linear8bitLt as LoraLinear8bitLt
from torch.nn import Linear


def get_float_weight(model: torch.nn.Module) -> torch.Tensor:
    """Get the float weight of a model.

    The model must expose ``in_features``, ``weight``, and optionally ``bias``
    (e.g. ``torch.nn.Linear``, ``Linear8bitLt``, ``Linear4bit``).

    Args:
        model: A linear-like module with ``in_features`` and ``weight``.

    Returns:
        The reconstructed float weight tensor.
    """
    device: torch.device = model.weight.device  # type: ignore[union-attr]
    in_features: int = model.in_features  # type: ignore[union-attr]
    with torch.no_grad():
        I = torch.eye(in_features).to(device)
        w = model(I)
        if hasattr(model, "bias") and isinstance(model.bias, torch.Tensor):
            w -= model.bias
        w = torch.transpose(w, 0, 1)
    w.requires_grad = bool(model.weight.requires_grad)
    return w


def replace_module_with_linear(model: torch.nn.Module, target: type) -> None:
    """Replace all child modules of ``target`` type with plain ``nn.Linear`` layers.

    Args:
        model: The model whose children will be inspected.
        target: The module class to replace.
    """
    for name, module in model.named_children():
        if isinstance(module, target):
            in_features: int = module.in_features  # type: ignore[union-attr]
            out_features: int = module.out_features  # type: ignore[union-attr]
            bias: bool = module.bias is not None
            new_module = torch.nn.Linear(in_features, out_features, bias)
            with torch.no_grad():
                new_module.weight.data = get_float_weight(module).data
                if bias:
                    new_module.bias.data = module.bias  # type: ignore[union-attr]
            setattr(model, name, new_module)
        else:
            replace_module_with_linear(module, target)


def dequantize(model: torch.nn.Module, dtype: str) -> None:
    """Dequantize a model by replacing quantized linear layers with float ones.

    Args:
        model: The model to dequantize.
        dtype: Quantization dtype — ``"int8"`` or ``"nf4"``.
    """
    target: type | None = None
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
    w = get_float_weight(model=m_int8)
    print(linear.weight)
    print(w)

    print("nf4 quant==============================================")
    base_linear_nf4 = Linear(config.in_features, config.out_features)
    base_linear_nf4.load_state_dict(linear.state_dict())
    m_nf4 = Linear4bit(base_layer=base_linear_nf4, adapter_name="default")
    m_nf4.to(config.device)
    print(m_nf4.weight)
    print(linear.weight)
    w = get_float_weight(model=m_nf4)
    print(w)


if __name__ == "__main__":
    main()
