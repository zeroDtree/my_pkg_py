import torch
from accelerate import Accelerator
from torch.nn.parallel import DistributedDataParallel

from ls_mlkit.pipeline.callback import BaseCallback, CallbackEvent
from ls_mlkit.pipeline.pipeline import TrainingState
from ls_mlkit.util.base_class.base_gm_class import BaseGenerativeModel, GMHookHandler


class TrainingStepCallback(BaseCallback):
    def __init__(self, training_handler: list[GMHookHandler]):
        super().__init__()
        self.training_handlers: list[GMHookHandler] = training_handler

    def on_event(self, event: CallbackEvent, *args, **kwargs):
        if event == CallbackEvent.STEP_START:
            self.step_start(*args, **kwargs)

    def step_start(self, *args, **kwargs):
        training_state: TrainingState = kwargs.get("training_state", None)
        diffuser: BaseGenerativeModel = kwargs.get("model", None)
        accelerator: Accelerator = kwargs.get("accelerator", None)
        if isinstance(diffuser, DistributedDataParallel):
            diffuser = diffuser.module
        assert training_state is not None
        assert diffuser is not None
        training_state.current_global_step
        prob_drop = torch.rand(1)
        if prob_drop < 5.0:
            for handler in self.training_handlers:
                handler.disable()
                # print(f"disable {handler.hook.name}")
        else:
            for handler in self.training_handlers:
                handler.enable()
                # print(f"enable {handler.hook.name}")
