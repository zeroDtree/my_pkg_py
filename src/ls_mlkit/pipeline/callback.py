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
    PRE_SAVE = "pre_save"
    POST_SAVE = "post_save"
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"
    # optimize
    PRE_COMPUTE_LOSS = "pre_compute_loss"
    POST_COMPUTE_LOSS = "post_compute_loss"
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"
    PRE_OPTIMIZER_STEP = "pre_optimizer_step"
    POST_OPTIMIZER_STEP = "post_optimizer_step"
    # eval
    PRE_EVAL = "pre_eval"
    POST_EVAL = "post_eval"
    PRE_EVAL_STEP = "pre_eval_step"
    POST_EVAL_STEP = "post_eval_step"


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
