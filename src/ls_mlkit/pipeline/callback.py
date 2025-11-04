from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List


class CallbackEvent(Enum):
    # training
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    # save, load
    BEFORE_SAVE = "before_save"
    AFTER_SAVE = "after_save"
    BEFORE_LOAD = "before_load"
    AFTER_LOAD = "after_load"
    # optimize
    BEFORE_COMPUTE_LOSS = "before_compute_loss"
    AFTER_COMPUTE_LOSS = "after_compute_loss"
    BEFORE_BACKWARD = "before_backward"
    AFTER_BACKWARD = "after_backward"
    BEFORE_OPTIMIZER_STEP = "before_optimizer_step"
    AFTER_OPTIMIZER_STEP = "after_optimizer_step"
    # eval
    BEFORE_EVAL = "before_eval"
    AFTER_EVAL = "after_eval"
    BEFORE_EVAL_STEP = "before_eval_step"
    AFTER_EVAL_STEP = "after_eval_step"


class BaseCallback(metaclass=ABCMeta):
    @abstractmethod
    def on_event(self, event: CallbackEvent, *args, **kwargs):
        """On event

        Args:
            event (CallbackEvent): the event to trigger
            *args: the arguments to pass to the callback
            **kwargs: the keyword arguments to pass to the callback
        """


class CallbackManager:
    def __init__(self):
        self.callbacks: List[BaseCallback] = []

    def add_callback(self, callback: BaseCallback):
        """Add a callback

        Args:
            callback (BaseCallback): the callback to add
        """
        if callback is not None:
            self.callbacks.append(callback)

    def add_callbacks(self, callbacks: List[BaseCallback]):
        """Add a list of callbacks

        Args:
            callbacks (List[BaseCallback]): the callbacks to add
        """
        if callbacks is not None and len(callbacks) > 0:
            self.callbacks.extend(callbacks)

    def trigger(self, event: CallbackEvent, *args, **kwargs):
        """Trigger all callbacks for a given event

        Args:
            event (CallbackEvent): the event to trigger
            *args: the arguments to pass to the callback
            **kwargs: the keyword arguments to pass to the callback
        """
        for callback in self.callbacks:
            callback.on_event(event, *args, **kwargs)
