import os

import psutil
import torch


def get_split_num(origin_type: str = "bf16", quant_type: str = "int8"):
    """
    Calculates the ratio of original type size to quantized type size.

    Args:
        origin_type (str, optional): The data type of the original tensor. Defaults to "bf16".
                                     Options are "fp32" and "bf16".
        quant_type (str, optional): The data type of the quantized tensor. Defaults to "int8".
                                    Options are "int8" and "nf4".

    Raises:
        ValueError: If the origin_type is not "fp32" or "bf16".
        ValueError: If the quant_type is not "int8" or "nf4".

    Returns:
        int: The ratio of the original type size to the quantized type size.
    """
    n_origin_bits = 16
    n_quant_bits = 8
    match origin_type:
        case "fp32":
            n_origin_bits = 32
        case "bf16":
            n_origin_bits = 16
        case _:
            raise ValueError("Wrong dtype")
    match quant_type:
        case "int8":
            n_quant_bits = 8
        case "nf4":
            n_quant_bits = 4
        case _:
            raise ValueError("Wrong dtype")
    return n_origin_bits // n_quant_bits


def print_cpu_memory():
    mem = psutil.virtual_memory()
    total = str(round(mem.total / 1024**3))
    used = str(round(mem.used / 1024**3))
    use_per = str(round(mem.percent))
    free = str(round(mem.free / 1024**3))
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_in_bytes = memory_info.rss
    print(f"CPU memory total: {total}GB, used: {used}GB ({use_per}%), free: {free}GB")
    print(f"CPU memory used by this process: {memory_usage_in_bytes / 1024**3:.2f}GB")


def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"GPU Allocated Memory: {allocated_memory:.2f} GB")
    print(f"GPU max_memory_allocated {max_allocated / (1024**3)} GB")


def show_gpu_and_cpu_memory():
    print_gpu_memory()
    print_cpu_memory()


def get_module_list(model, no_split_module_classes=None):
    """
    Return the full dotted names of modules at the split boundary.
    Recursion stops at any module whose class name is in
    `no_split_module_classes`, or at leaf modules.
    """
    if no_split_module_classes is None:
        no_split_module_classes = []

    module_list = []

    def _get_module_list(module: torch.nn.Module, parent_name=""):
        if module.__class__.__name__ in no_split_module_classes:
            module_list.append(parent_name)
            return
        if next(module.named_children(), None) is None:
            module_list.append(parent_name)
            return
        for name, sub_module in module.named_children():
            extend_name = f"{parent_name}.{name}" if parent_name else name
            _get_module_list(sub_module, extend_name)

    _get_module_list(model)
    return module_list


def uniformly_split(lst: list, n: int) -> list[list]:
    """Split a list into n roughly equal chunks using pure Python."""
    k, rem = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < rem else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


def get_partition_block(module_list: list, num_block: int) -> dict:
    """Partition module_list into num_block groups and return per-module metadata."""
    num_block = min(num_block, len(module_list))
    module_groups = uniformly_split(module_list, num_block)
    module_info: dict = {}
    for i, group in enumerate(module_groups):
        first_block = i == 0
        last_block = i == (num_block - 1)
        n_module = len(group)
        for j, module_name in enumerate(group):
            module_info[module_name] = {
                "first_block_flag": first_block,
                "last_block_flag": last_block,
                "first_module_flag": j == 0,
                "last_module_flag": j == (n_module - 1),
            }
    return module_info
