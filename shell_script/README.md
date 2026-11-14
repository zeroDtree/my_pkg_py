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

## Run Python

GPU training or eval. The launcher exits when no GPU is visible. It selects `configs/accelerate/acc_cfg_<N>cards.yaml` for N in 1, 2, 3, 4, or 8, and passes that same N as `--num_processes`. The script exits with the training process exit code.

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
