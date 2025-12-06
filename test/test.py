import wandb

wandb.init(project="test")

for i in range(10):
    wandb.log({"a": i}, step=i)
