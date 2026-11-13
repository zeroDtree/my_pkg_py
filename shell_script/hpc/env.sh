#!/usr/bin/env bash
# Shared Slurm/CUDA bootstrap. Source from a batch script (do not execute).
#
# Slurm often copies the batch script into SpoolDir and runs that copy.
# Do NOT derive the project root from BASH_SOURCE — use SLURM_SUBMIT_DIR.
#
# Requires: SLURM_SUBMIT_DIR set by Slurm (submit from the repo root).
# Optional: CUDA_HOME (default: $HOME/software/cuda/cuda-12.4).
# Change the default for a different cluster toolkit.

if [[ -z "${SLURM_SUBMIT_DIR:-}" || ! -d "${SLURM_SUBMIT_DIR}" ]]; then
	echo "ERROR: SLURM_SUBMIT_DIR is unset or invalid: '${SLURM_SUBMIT_DIR:-}'" >&2
	echo "Submit with: sbatch shell_script/hpc/<job>.slurm  (from the repo root)" >&2
	return 1 2>/dev/null || exit 1
fi

export PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}" || {
	echo "ERROR: cannot cd to PROJECT_ROOT=${PROJECT_ROOT}" >&2
	return 1 2>/dev/null || exit 1
}

CUDA_HOME="${CUDA_HOME:-${HOME}/software/cuda/cuda-12.4}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "===== HPC env ====="
echo "Hostname            : $(hostname)"
echo "PROJECT_ROOT        : ${PROJECT_ROOT}"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "CUDA_HOME           : ${CUDA_HOME}"
echo
