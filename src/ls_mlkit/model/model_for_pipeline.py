from typing import Any

import torch
from torch.nn import Module

from ..util.hook.model_hook import ModelHook, ModelHookHandler, ModelHookManager, ModelHookStageType


class ModelForPipeline(Module):

    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.hook_manager = ModelHookManager()

    def get_model_device(self) -> torch.device:
        model_device = next(self.model.parameters()).device
        return model_device

    def forward(
        self,
        **batch: dict[str, Any],
    ) -> dict:
        model = self.model
        self.hook_manager.run_hooks(stage=ModelHookStageType.PRE_COMPUTE_LOSS, model=model, batch=batch)
        model_output = model(**batch)
        self.hook_manager.run_hooks(
            stage=ModelHookStageType.POST_COMPUTE_LOSS, model=model, batch=batch, model_output=model_output
        )
        return model_output

    def register_hooks(self, hooks: list[ModelHook]) -> list[ModelHookHandler]:
        return self.hook_manager.register_hooks(hooks)
