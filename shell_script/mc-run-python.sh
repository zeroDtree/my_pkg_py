#!/usr/bin/env bash

# @help-begin
# GPU-only Accelerate launcher. Requires at least one visible GPU
# (CUDA_VISIBLE_DEVICES or nvidia-smi). For CPU, use run-python.sh.
#
# Usage:
#   shell_script/mc-run-python.sh SCRIPT [args...]
#
# Picks configs/accelerate/acc_cfg_<N>cards.yaml for the visible GPU count.
# Available counts: 1, 2, 3, 4, 8.
#
# Env: MC_SKIP_UV_SYNC — set to 1 to skip dependency synchronization.
# Env: CUDA_VISIBLE_DEVICES — comma-separated GPU identifiers.
# Env: MC_ENABLE_PROXY — set to 1 before this script to enable the HTTP proxy.
#
# Under Slurm (SLURM_JOB_ID set), MC_SKIP_UV_SYNC defaults to 1.
# Example:
#   bash shell_script/mc-run-python.sh example/butterfly_edm/main.py
# @help-end

# @help-options-begin
#   -h, --help              show help
# @help-options-end

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
	awk '/^# @help-begin$/{f=1; next} /^# @help-end$/{f=0} f' "$0"
	printf '%s\n' '#' 'Options:' '#'
	awk '/^# @help-options-begin$/{f=1; next} /^# @help-options-end$/{f=0} f' "$0"
	exit 0
}

[[ $# -ge 1 ]] || usage
case "${1:-}" in
	-h|--help) usage ;;
esac

source "$SCRIPT_DIR/prepare.sh"
cd "$PROJECT_ROOT" || exit 1

# Compute nodes often lack network/proxy; sync on the login node before sbatch.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	MC_SKIP_UV_SYNC="${MC_SKIP_UV_SYNC:-1}"
fi

count_visible_gpus() {
	if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
		nvidia-smi -L 2>/dev/null | wc -l
	else
		local IFS=','
		read -ra ids <<< "$CUDA_VISIBLE_DEVICES"
		echo "${#ids[@]}"
	fi
}

append_command_log() {
	local runner="$1"
	shift
	local log_line="cuda${CUDA_VISIBLE_DEVICES:+ $CUDA_VISIBLE_DEVICES} bash $runner $*"
	local log_dir="$PROJECT_ROOT/artifacts/command_logs"
	local entry_script="${1:-unknown}"
	local log_name="${entry_script//\//_}"
	log_name="${log_name//./_}"
	local log_file="$log_dir/${log_name}.log"

	mkdir -p "$log_dir"

	if [[ -f "$log_file" ]]; then
		local last_line
		last_line="$(tail -n 1 "$log_file")"
		if [[ "$last_line" == "$log_line" ]]; then
			return 0
		fi
	fi

	printf '%s\n' "$log_line" >> "$log_file"
}

n_cards="$(count_visible_gpus)"
n_cards="${n_cards// /}"

if [[ -z "$n_cards" || "$n_cards" -lt 1 ]]; then
	echo "Error: mc-run-python.sh is GPU-only (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
	echo "Use shell_script/run-python.sh for CPU runs." >&2
	exit 1
fi

config_file="configs/accelerate/acc_cfg_${n_cards}cards.yaml"
if [[ ! -f "$config_file" ]]; then
	echo "Error: missing $config_file for $n_cards visible GPU(s)." >&2
	echo "Available card counts: 1, 2, 3, 4, 8." >&2
	exit 1
fi

append_command_log "$0" "$@"

entry_script="$1"
shift

echo "n_cards: $n_cards"
echo "script: $entry_script"
echo "args: $*"
echo "config_file: $config_file"

if [[ "${MC_SKIP_UV_SYNC:-}" != "1" ]]; then
	uv sync
fi

set +e
uv run python -m accelerate.commands.launch \
	--main_process_port "$(shuf -i 10000-60000 -n 1)" \
	--config_file "$config_file" \
	--num_processes "$n_cards" \
	"$entry_script" \
	"$@"
exit_code=$?
set -e

# Notify only on successful completion; skip email for manual interrupts (debug).
if [[ "$exit_code" -eq 0 ]]; then
	command_args=(cuda)
	if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
		command_args+=("$CUDA_VISIBLE_DEVICES")
	fi
	command_args+=(bash "$0" "$entry_script" "$@")
	bash "$SCRIPT_DIR/email.sh" "$exit_code" -- "${command_args[@]}" || true
fi
exit "$exit_code"
