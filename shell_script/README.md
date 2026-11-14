# Shell scripts

Invoke as `bash shell_script/...` from the project root. Scripts then `cd` to the repo via `BASH_SOURCE`.

Training artifacts are relative to that directory, so these launchers write `<repo>/artifacts/`. Starting the same example from inside `example/<name>/` writes `example/<name>/artifacts/` instead. Resume a run from the directory where it was started.

## Post-clone check

```bash
bash shell_script/init-check.sh
```

This syncs dependencies, then runs pytest, ruff, and ty. It installs the nbstripout git filter when this directory is a git work tree. Set `MC_ENABLE_PROXY=1` before the script to enable the HTTP proxy for `uv sync`.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MC_ENABLE_PROXY` | off | Set to `1` before a runner to enable the HTTP proxy. |
| `PROXY_PORT_FILE` | `$HOME/.config/clash-proxy-port` | Port file. The first line is a port from 1 to 65535; the proxy URL is `http://127.0.0.1:<port>`. `proxy_on` fails if the file is missing or invalid. |
| `MC_SKIP_UV_SYNC` | off | Set to `1` to skip `uv sync`. Under Slurm (`SLURM_JOB_ID` set), defaults to `1`. |
| `CUDA_VISIBLE_DEVICES` | unset | Comma-separated GPU ids. `mc-run-python.sh` is GPU-only and uses this, or `nvidia-smi -L`, to pick a card-count config. |
| `FROM_NAME` | `mlkit` | Display name on the completion email. SMTP settings live in `env_email.sh`. |

## Run Python

GPU training or eval. The launcher exits when no GPU is visible. It selects `configs/accelerate/acc_cfg_<N>cards.yaml` for N in 1, 2, 3, 4, or 8, and passes that same N as `--num_processes`. On exit code 0 it sends a completion email. The message includes a replayable command, `cuda <GPU ids> bash shell_script/mc-run-python.sh <script> [args...]`. GPU ids are omitted when `CUDA_VISIBLE_DEVICES` is unset. A missing `env_email.sh` does not change the training exit code.

```bash
export MC_ENABLE_PROXY=1
bash shell_script/mc-run-python.sh example/butterfly_edm/main.py
```

Write the proxy port into the port file before enabling the proxy.

CPU, or any run that should not use Accelerate:

```bash
bash shell_script/run-python.sh example/butterfly_edm/main.py
bash shell_script/run-python.sh -m pytest
```

Commands are appended under `artifacts/command_logs/`.

## Email

Copy the example and fill in SMTP settings (`FROM_EMAIL`, `FROM_SMTP_HOST`, `FROM_SMTP_PORT`, `FROM_SMTP_PASSWORD`, `TO_EMAIL`). `env_email.sh` is gitignored.

```bash
cp shell_script/env_email.sh.example shell_script/env_email.sh
bash shell_script/email.sh 0
bash shell_script/email.sh 0 -- example/butterfly_edm/main.py
```

## HPC / Slurm

Submit from the repo root. `hpc/env.sh` sets `PROJECT_ROOT` from `SLURM_SUBMIT_DIR`, not `BASH_SOURCE`.

`CUDA_HOME` defaults to `$HOME/software/cuda/cuda-12.4`. `#SBATCH --partition=phys_hq` is this site's partition. Change both for another cluster.

| Script | Role |
| --- | --- |
| `hpc/test.slurm` | Node smoke test: driver and toolkit. The PyTorch matmul runs only when `torch` imports. |
| `hpc/run.slurm` | Sets the cluster env, then calls `mc-run-python.sh`. |

Sync the venv on the login node before submitting. Under Slurm, `MC_SKIP_UV_SYNC` defaults to `1`.

```bash
mkdir -p artifacts/logs/slurm
sbatch shell_script/hpc/test.slurm
sbatch shell_script/hpc/run.slurm -- example/butterfly_edm/main.py
```
