import torch
from ls_mlkit.util.cuda import check_cuda

check_cuda()

x = torch.randn(5)

x = x.to("cuda")

print(x)
