from tqdm.auto import tqdm
import time
import torch

model = torch.nn.Linear(10, 10)
model = model.to("cuda")

from accelerate import Accelerator
accelerator = Accelerator()
model = accelerator.prepare(model)

print(type(model))

model = accelerator.prepare(model)

print(type(model))