"""Smoke tests for ls_mlkit.util.offload (CUDA required for full coverage)."""

import unittest

import torch
import torch.nn as nn

from ls_mlkit.util.offload import ModelOffloadHookContext, OffloadContext


class TestOffloadSmoke(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_model_offload_context_forward(self) -> None:
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        ).cuda()
        with ModelOffloadHookContext(model=model, num_block=2, enable=True):
            x = torch.randn(3, 4, device="cuda")
            y = model(x)
            self.assertEqual(y.shape, (3, 2))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_offload_context_forward_backward(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)).cuda()
        named_grads: dict = {}
        with OffloadContext(model=model, named_grads=named_grads, num_block=2):
            x = torch.randn(2, 4, device="cuda", requires_grad=True)
            y = model(x)
            y.sum().backward()
        self.assertGreater(len(named_grads), 0)

    def test_forward_backward_unsupported_strategy(self) -> None:
        from ls_mlkit.util.offload.forward_backward_offload import ForwardBackwardOffloadHookContext

        model = nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            ForwardBackwardOffloadHookContext(model=model, enable=True, strategy="module")


if __name__ == "__main__":
    unittest.main()
