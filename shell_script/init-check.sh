#!/usr/bin/env bash

# @help-begin
# Install dependencies and run the post-init checks from the project root.
#
# Usage:
#   bash shell_script/init-check.sh
#
# Syncs the dev group, then runs pytest, ruff, and ty.
# Installs the nbstripout git filter when this directory is a git work tree.
# Outside a work tree, that install is skipped.
#
# Env: MC_ENABLE_PROXY — set to 1 before this script to enable the HTTP proxy.
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

case "${1:-}" in
	-h|--help) usage ;;
	"") ;;
	*)
		echo "Error: unknown argument: $1" >&2
		exit 1
		;;
esac

source "$SCRIPT_DIR/prepare.sh"
cd "$PROJECT_ROOT" || exit 1

uv sync
uv run pytest
uv run ruff check
uv run ty check

project_root_physical="$(cd "$PROJECT_ROOT" && pwd -P)"
git_root="$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$git_root" ]]; then
	git_root="$(cd "$git_root" && pwd -P)"
fi
if [[ -n "$git_root" && "$git_root" == "$project_root_physical" ]]; then
	uv run nbstripout --install --attributes .gitattributes
else
	echo "Skipping nbstripout install: not a git work tree." >&2
fi
