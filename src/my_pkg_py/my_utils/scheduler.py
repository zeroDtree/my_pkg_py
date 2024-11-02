import math


def cosine_decay_with_warmup(value, current, total, warmup_steps=0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        return (
            value
            * (
                1
                + math.cos(math.pi * (current - warmup_steps) / (total - warmup_steps))
            )
            / 2
        )


def linear_decay_with_warmup(value, current, total, warmup_steps=0):
    if current < warmup_steps:
        return value * current / warmup_steps
    else:
        return value * (1 - (current - warmup_steps) / (total - warmup_steps))


def constant(value, current, total, warmup_steps=0):
    return value


FUNCTION_MAPPING = {
    "cosine_decay_with_warmup": cosine_decay_with_warmup,
    "linear_decay_with_warmup": linear_decay_with_warmup,
    "constant": constant,
}


class Scheduler:
    def __init__(
        self,
        info,
        total,
        warmup_steps=None,
        warmup_ratio=0,
    ):
        self.info = info
        self.total = total
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.current = 0
        if self.warmup_steps is None:
            self.warmup_steps = int(self.total * self.warmup_ratio)
        for key, value in self.info.items():
            if value.get("value") is None:
                raise ValueError(f"value of {key} is not defined")
            if value.get("schedule") is None:
                value["schedule"] = FUNCTION_MAPPING["constant"]
            if value.get("warmup_steps") is None:
                value["warmup_steps"] = int(self.total * value["warmup_ratio"])

    def step(self):
        self.current += 1
        for key, value in self.info.items():
            value["current_value"] = value["schedule"](
                value["value"], self.current, self.total, value["warmup_steps"]
            )

    def get(self, key=None):
        if key is None:
            return self.info
        else:
            return self.info[key]["current_value"]


class ObjectAttrScheduler:
    def __init__(
        self,
        obj,
        attr_name,
        total=None,
        warmup_steps=None,
        warmup_ratio=0,
        strategy="constant",
    ):
        assert hasattr(obj, attr_name), f"{attr_name} is not an attribute of {obj}"
        self.obj = obj
        self.attr_name = attr_name
        self.strategy = strategy
        self.info = {
            attr_name: {
                "value": getattr(obj, attr_name),
                "schedule": FUNCTION_MAPPING[strategy],
                "warmup_steps": warmup_steps,
                "warmup_ratio": warmup_ratio,
            }
        }
        self.scheduler = Scheduler(self.info, total, warmup_steps, warmup_ratio)

    def step(self):
        self.scheduler.step()
        setattr(self.obj, self.attr_name, self.scheduler.get(self.attr_name))

    def get(self):
        return getattr(self.obj, self.attr_name)


class ObjectAttrsScheduler:
    def __init__(
        self,
        obj,
        attr_names,
        total=None,
        warmup_steps=None,
        warmup_ratio=0,
        strategy="constant",
    ):
        self.obj = obj
        self.attr_names = attr_names
        self.strategy = strategy
        self.info = {}
        for attr_name in attr_names:
            self.info.update(
                {
                    attr_name: {
                        "value": getattr(obj, attr_name),
                        "schedule": FUNCTION_MAPPING[strategy],
                        "warmup_steps": warmup_steps,
                        "warmup_ratio": warmup_ratio,
                    }
                }
            )
        self.scheduler = Scheduler(self.info, total, warmup_steps, warmup_ratio)

    def step(self):
        self.scheduler.step()
        for attr_name in self.attr_names:
            setattr(self.obj, attr_name, self.scheduler.get(attr_name))

    def get(self):
        return {attr_name: getattr(self.obj, attr_name) for attr_name in self.attr_names}


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

    test = Test(10)
    scheduler = ObjectAttrsScheduler(
        test,
        attr_names=["x", "y", "z"],
        total=total,
        warmup_ratio=warmup_ratio,
        strategy="cosine_decay_with_warmup",
    )
    for i in range(total):
        scheduler.step()
        wandb.log(scheduler.get())
    # scheduler_x = ObjectAttrScheduler(
    #     test,
    #     "x",
    #     total=total,
    #     warmup_ratio=warmup_ratio,
    #     strategy="cosine_decay_with_warmup",
    # )
    # scheduler_y = ObjectAttrScheduler(
    #     test,
    #     "y",
    #     total=total,
    #     warmup_ratio=warmup_ratio,
    #     strategy="linear_decay_with_warmup",
    # )
    # scheduler_z = ObjectAttrScheduler(
    #     test, "z", total=total, warmup_ratio=warmup_ratio, strategy="constant"
    # )
    # for i in range(total):
    #     scheduler_x.step()
    #     scheduler_y.step()
    #     scheduler_z.step()
    #     wandb.log({"x": test.x, "y": test.y, "z": test.z})
