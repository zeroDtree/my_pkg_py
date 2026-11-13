#!/usr/bin/env bash

# @help-begin
# Run a Python script from the project root with uv.
#
# Usage:
#   shell_script/run-python.sh SCRIPT [args...]
#
# Pass -m before a module name to run a module instead of a script.
#
# Env: MC_SKIP_UV_SYNC — set to 1 to skip dependency synchronization.
# Env: MC_ENABLE_PROXY — set to 1 before this script to enable the HTTP proxy.
#
# Under Slurm (SLURM_JOB_ID set), MC_SKIP_UV_SYNC defaults to 1.
# Example:
#   bash shell_script/run-python.sh example/butterfly_edm/main.py
#   bash shell_script/run-python.sh -m pytest
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
	MC_SKIP_UV_SYNC="${MC_SKIP_UV_SYNC:-1}"
fi

if [[ "${MC_SKIP_UV_SYNC:-}" != "1" ]]; then
	uv sync
fi

uv run python "$@"
