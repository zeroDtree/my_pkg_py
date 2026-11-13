#!/usr/bin/env bash

# @help-begin
# Send an experiment-completion email via Python smtplib (no mutt required).
#
# Usage:
#   bash shell_script/email.sh [exit_code] [-- command...]
#
# Env: credentials come from shell_script/env_email.sh (gitignored).
# Copy env_email.sh.example to env_email.sh and fill in SMTP settings.
#
# If no options are passed, the default behavior is equivalent to:
#   bash shell_script/email.sh 0
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
esac

cd "$PROJECT_ROOT"

ENV_EMAIL="$SCRIPT_DIR/env_email.sh"
if [[ ! -f "$ENV_EMAIL" ]]; then
	echo "Error: missing $ENV_EMAIL" >&2
	echo "Copy shell_script/env_email.sh.example to shell_script/env_email.sh and fill in SMTP settings." >&2
	exit 1
fi

# shellcheck source=env_email.sh
source "$ENV_EMAIL"

EXIT_CODE="${1:-0}"
if [[ $# -gt 0 ]]; then
	shift
fi
if [[ "${1:-}" == "--" ]]; then
	shift
fi

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
	PYTHON_BIN="python3"
fi

echo "Sending experiment completion notification..."
if "${PYTHON_BIN}" "${SCRIPT_DIR}/email_assets/send_experiment_email.py" "${EXIT_CODE}" -- "$@"; then
	:
else
	echo "Notification send failed." >&2
	exit 1
fi
