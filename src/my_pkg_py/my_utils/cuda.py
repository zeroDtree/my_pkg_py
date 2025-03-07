import torch


def check_cuda():
    print("============================================================")
    print("cuda.is_available:", torch.cuda.is_available())
    print("cuda.device_count:", torch.cuda.device_count())
    print("version.cuda:", torch.version.cuda)
    print("current_device:", torch.cuda.current_device())
    print(
        "cuda.get_device_name:", torch.cuda.get_device_name(torch.cuda.current_device())
    )
    print("torch.backends.cudnn.is_available", torch.backends.cudnn.is_available())
    print("print(torch.backends.cudnn.version", torch.backends.cudnn.version())
    print(
        "(free, total)GB", [x / (1024 * 1024 * 1024) for x in torch.cuda.mem_get_info()]
    )
    print(
        "torch.cuda.max_memory_allocated",
        torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),
        "GB",
    )
    current_backend = torch.backends.cuda.preferred_linalg_library()
    print(f"Current preferred linalg library: {current_backend}")
    print("============================================================")


if __name__ == "__main__":
    check_cuda()
