import torch
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import AdamW

x, c = make_moons(10, noise=0.15)

print(x)
print(c)


class MoonsClassifier(Module):
    def __init__(self, dim: int = 2, h=64, n_labels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, n_labels),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


model = MoonsClassifier()
optimizer = AdamW(model.parameters())

for i in range(2000):
    x, c = make_moons(256, noise=0.15)
    x = Tensor(x)
    c = Tensor(c).long()
    logits = model(x)
    loss = F.cross_entropy(logits, c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def classifier_condition_loss(x: Tensor, c: Tensor, model: Tensor):
    x = x.detach().clone()
    logits = model(x)
    loss = F.cross_entropy(logits, c)
    grad = torch.autograd.grad(loss, x)
    return -grad
