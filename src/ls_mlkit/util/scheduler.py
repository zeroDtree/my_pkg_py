import math
from enum import Enum
from typing import Any, Callable


def cosine_decay_with_warmup(value, current, total, warmup_steps=0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        return value * (1 + math.cos(math.pi * (current - warmup_steps) / (total - warmup_steps))) / 2


def linear_decay_with_warmup(value, current, total, warmup_steps=0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        return value * (1 - (current - warmup_steps) / (total - warmup_steps))


def constant_with_warmup(value, current, total, warmup_steps=0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        return value


def exponential_decay_with_warmup(value, current, total, warmup_steps=0, decay_rate=5.0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        progress = (current - warmup_steps) / (total - warmup_steps)
        return value * math.exp(-decay_rate * progress)


class SchedulerType(Enum):
    COSINE_DECAY_WITH_WARMUP = "cosine_decay_with_warmup"
    LINEAR_DECAY_WITH_WARMUP = "linear_decay_with_warmup"
    EXPONENTIAL_DECAY_WITH_WARMUP = "exponential_decay_with_warmup"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


FUNCTION_MAPPING = {
    SchedulerType.COSINE_DECAY_WITH_WARMUP: cosine_decay_with_warmup,
    SchedulerType.LINEAR_DECAY_WITH_WARMUP: linear_decay_with_warmup,
    SchedulerType.EXPONENTIAL_DECAY_WITH_WARMUP: exponential_decay_with_warmup,
    SchedulerType.CONSTANT_WITH_WARMUP: constant_with_warmup,
}


class Scheduler:
    def __init__(
        self,
        info: dict[str, dict[str, Any]],
        total: int,
    ):
        self.info = info
        self.total = total
        self.current = 0
        for key, value in self.info.items():
            if value.get("value") is None:
                raise ValueError(f"value of {key} is not defined")
            if value.get("schedule") is None:
                raise ValueError(f"schedule of {key} is not defined")
            if value.get("warmup_steps") is None:
                assert (
                    value.get("warmup_ratio") is not None
                ), f"warmup_ratio of {key} must be provided if warmup_steps is not provided"
                value["warmup_steps"] = int(self.total * value["warmup_ratio"])

    def step(self):
        """Step the scheduler"""
        self.current += 1
        for key, value in self.info.items():
            value["current_value"] = value["schedule"](value["value"], self.current, self.total, value["warmup_steps"])

    def get(self, key=None):
        """Get the current value of the scheduler

        Args:
            key (str, optional): The key of the scheduler to get. If None, return the entire scheduler info. Defaults to None.

        Returns:
            dict[str, Any] or Any: The entire scheduler info or the value of the scheduler for the given key
        """
        if key is None:
            return self.info
        else:
            return self.info[key]["current_value"]


class ObjectAttrsScheduler:
    def __init__(
        self,
        obj: object,
        attr_names: list[str],
        total: int,
        warmup_steps: int = None,
        warmup_ratio: float = 0,
        strategy: SchedulerType = SchedulerType.CONSTANT_WITH_WARMUP,
        setter_methods: dict[str, Callable] = None,
        getter_methods: dict[str, Callable] = None,
    ):
        self.obj = obj
        self.attr_names = attr_names
        self.strategy = strategy
        self.setter_methods = setter_methods or {}
        self.getter_methods = getter_methods or {}

        self.info = {}
        for attr_name in attr_names:
            # Get initial value
            getter_method = self.getter_methods.get(attr_name)
            if getter_method is not None:
                assert hasattr(obj, getter_method), f"{getter_method} is not a method of {obj}"
                initial_value = getattr(obj, getter_method)()
            else:
                assert hasattr(obj, attr_name), f"{attr_name} is not an attribute of {obj}"
                initial_value = getattr(obj, attr_name)

            # Validate setter if provided
            setter_method = self.setter_methods.get(attr_name)
            if setter_method is not None:
                assert hasattr(obj, setter_method), f"{setter_method} is not a method of {obj}"

            self.info.update(
                {
                    attr_name: {
                        "value": initial_value,
                        "schedule": FUNCTION_MAPPING[strategy],
                        "warmup_steps": warmup_steps,
                        "warmup_ratio": warmup_ratio,
                    }
                }
            )
        self.scheduler = Scheduler(self.info, total)

    def step(self):
        self.scheduler.step()
        for attr_name in self.attr_names:
            new_value = self.scheduler.get(attr_name)

            # Use custom setter if provided, otherwise use setattr
            setter_method = self.setter_methods.get(attr_name)
            if setter_method is not None:
                getattr(self.obj, setter_method)(new_value)
            else:
                setattr(self.obj, attr_name, new_value)

    def get(self):
        result = {}
        for attr_name in self.attr_names:
            # Use custom getter if provided, otherwise use getattr
            getter_method = self.getter_methods.get(attr_name)
            if getter_method is not None:
                result[attr_name] = getattr(self.obj, getter_method)()
            else:
                result[attr_name] = getattr(self.obj, attr_name)
        return result


if __name__ == "__main__":
    import wandb

    wandb.init(project="scheduler-test")
    total = 100
    warmup_ratio = 0.1

    class Test:
        def __init__(self, value):
            self.x = value
            self.y = value
            self.z = value

        def set_x(self, value):
            self.x = value

        def get_x(self):
            return self.x

    test = Test(10)
    scheduler = ObjectAttrsScheduler(
        test,
        attr_names=["x", "y", "z"],
        total=total,
        warmup_ratio=warmup_ratio,
        strategy=SchedulerType.EXPONENTIAL_DECAY_WITH_WARMUP,
        setter_methods={"x": "set_x"},
        getter_methods={"x": "get_x"},
    )
    for i in range(total):
        scheduler.step()
        wandb.log(scheduler.get(), step=i)
