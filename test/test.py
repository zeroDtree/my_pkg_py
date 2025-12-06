import wandb


wandb.init(project="test")

for i in range(10):
    if i % 2 == 0:
        wandb.log({"b": i}, step=i)
    wandb.log({"a": i}, step=i)
