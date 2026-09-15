from typing import List

import torch
from peft import LoraConfig, get_peft_model


def find_linear_modules(model) -> List[str]:
    """Find the linear modules in a model

    Args:
        model (torch.nn.Module): the model to find the linear modules

    Returns:
        List[str]: the names of the linear modules
    """
    linear_cls = torch.nn.Linear
    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any([output_layer in name for output_layer in output_layer_names]):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def get_lora_model(model, lora_config):
    """Get a LoRA model

    Args:
        model (torch.nn.Module): the model to get the LoRA model
        lora_config (LoraConfig): the LoRA configuration

    Returns:
        torch.nn.Module: the LoRA model
    """
    taget_modules = find_linear_modules(model)
    lora_config = LoraConfig(
        r=lora_config["lora_r"],
        target_modules=taget_modules,
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
    )
    model = get_peft_model(model, lora_config)
    return model
