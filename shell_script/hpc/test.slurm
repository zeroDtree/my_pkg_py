#!/usr/bin/env bash
# Change --partition for your cluster. phys_hq is this site's default.
#SBATCH --job-name=slurm-test
#SBATCH --partition=phys_hq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/logs/slurm/slurm-test-%j.out
#SBATCH --error=artifacts/logs/slurm/slurm-test-%j.err

# Smoke-test SLURM job: node / CUDA / optional PyTorch (project uv env).
# Submit from the repo root; logs go under artifacts/logs/slurm/.
#
# Usage:
#   mkdir -p artifacts/logs/slurm
#   sbatch shell_script/hpc/test.slurm
#   squeue -u "$USER"
#   cat artifacts/logs/slurm/slurm-test-<jobid>.out
#
# Optional overrides at submit time:
#   sbatch --export=ALL,CUDA_HOME=$HOME/software/cuda/cuda-12.4 shell_script/hpc/test.slurm

set -euo pipefail

# Slurm may run a spool copy of this script; resolve paths via SLURM_SUBMIT_DIR.
source "${SLURM_SUBMIT_DIR}/shell_script/hpc/env.sh"

echo "===== SLURM smoke test ====="
echo "Date          : $(date)"
echo "Working dir   : $(pwd)"
echo "User          : $(whoami)"
echo
echo "===== Job environment ====="
echo "SLURM_JOB_NODELIST : ${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_ON_NODE : ${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_GPUS_ON_NODE : ${SLURM_GPUS_ON_NODE:-}"
echo
echo "===== CPU info ====="
nproc
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket' || true
echo
echo "===== GPU / driver ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo
echo "===== CUDA toolkit ====="
if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
  nvcc --version
else
  echo "nvcc not found under CUDA_HOME=${CUDA_HOME}"
  exit 1
fi
echo
echo "===== Memory ====="
free -h || true
echo
echo "===== PyTorch (uv project env) ====="
if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  echo "Missing project venv: ${PROJECT_ROOT}/.venv"
  echo "Create it first with: uv sync"
  exit 1
fi

if ! uv run python -c "import torch" >/dev/null 2>&1; then
  echo "torch is not installed; skipping CUDA matmul"
  echo
  echo "===== Done ====="
  exit 0
fi

uv run python - <<'PY'
import sys
import torch

print(f"python          : {sys.executable}")
print(f"torch           : {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda.is_available : {torch.cuda.is_available()}")
print(f"device_count    : {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch.cuda.is_available() is False")

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"device[{i}]       : {torch.cuda.get_device_name(i)}")
    print(f"  capability    : {props.major}.{props.minor}")
    print(f"  total_memory  : {props.total_memory / (1024**3):.2f} GiB")

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"matmul ok       : {tuple(y.shape)} dtype={y.dtype}")
print("PASS: CUDA + PyTorch")
PY

echo
echo "===== Done ====="
