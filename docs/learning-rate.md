# Learning rate schedule length

`get_learing_rate_scheduler` builds the scheduler with `n_training_steps * accelerator.num_processes`. The value stored in `cfg.train.n_steps` is only the per-rank loop count from `get_total_training_steps`.

The schedule still finishes at the end of training. Two Accelerate defaults make the multiplier necessary.

`Accelerator.prepare` wraps the scheduler in `AcceleratedScheduler`. This project leaves `split_batches` at its default, `False`. Each call to the wrapped `scheduler.step()` then advances the inner scheduler `num_processes` times, while gradients are synced. Example configs set `gradient_accumulation_steps` to 1, and the training loop does not change Accelerate's sync flag, so every call takes that path.

`prepare` also shards the dataloader with `BatchSamplerShard`. Each rank's epoch length is

\[
\left\lceil \left\lceil N / P \right\rceil / B \right\rceil
\]

which is about \(1/P\) of the single-process batch count. \(N\) is the dataset length, \(P\) is `num_processes`, and \(B\) is the per-device batch size. Inner steps over the run are the per-rank loop count times \(P\), which is the single-process batch count.

For `epochs`, `get_total_training_steps` returns `n_epochs * steps_per_epoch`. Multiplying by `num_processes` matches \(\lceil N \times n_{\mathrm{epochs}} / B \rceil\), up to the double ceiling above.

For `steps`, each rank runs the configured `n_steps`. The scheduler length is `n_steps * num_processes`.

Logged `n_steps` is the per-rank loop length. The scheduler uses that length times the process count. On one process the multiplier is 1.
