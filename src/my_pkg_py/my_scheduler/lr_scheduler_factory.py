import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_lambda_lr_scheduler(
    optimizer, num_warmup_steps, num_training_steps, lr_scheduler_type="linear"
):
    def cosine_lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # decay learning rate
            return (
                math.cos(
                    (current_step - num_warmup_steps)
                    / (num_training_steps - num_warmup_steps)
                    * math.pi
                )
                + 1
            ) / 2

    def linear_lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # decay learning rate
            return 1 - (
                (current_step - num_warmup_steps)
                / (num_training_steps - num_warmup_steps)
            )

    def constant_lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # constant learning rate
            return 1.0

    match lr_scheduler_type:
        case "cosine":
            lr_lambda = cosine_lr_lambda
        case "linear":
            lr_lambda = linear_lr_lambda
        case "constant":
            lr_lambda = constant_lr_lambda
    return LambdaLR(optimizer, lr_lambda)


def get_lr_scheduler(
    optimizer, num_warmup_steps, num_training_steps, lr_scheduler_type="linear"
):
    if lr_scheduler_type in ["cosine", "linear", "constant"]:
        return get_lambda_lr_scheduler(
            optimizer, num_warmup_steps, num_training_steps, lr_scheduler_type
        )
    elif lr_scheduler_type in ["cosine_annealing"]:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_training_steps)
