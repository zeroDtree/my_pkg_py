#!/usr/bin/env bash
set -e

# One port on the first line. PROXY_PORT_FILE overrides the path.
# When unset, use the file clash-tunnel.sh writes by default.
# A missing or invalid file makes proxy_on fail.
__proxy_expand_path() {
	local path="$1"
	case "$path" in
	'~')
		printf '%s\n' "$HOME"
		;;
	'~/'*)
		printf '%s\n' "$HOME/${path:2}"
		;;
	*)
		printf '%s\n' "$path"
		;;
	esac
}

__proxy_port_file() {
	if [[ -n "${PROXY_PORT_FILE:-}" ]]; then
		__proxy_expand_path "$PROXY_PORT_FILE"
	else
		printf '%s\n' "$HOME/.config/clash-proxy-port"
	fi
}

__proxy_resolve() {
	local path raw
	path="$(__proxy_port_file)"
	if [[ ! -f "$path" ]]; then
		echo "proxy_on: $path does not exist" >&2
		echo "Create this file and write the proxy port in it, or run clash-tunnel.sh on the machine that runs Clash." >&2
		return 1
	fi
	if [[ ! -r "$path" ]]; then
		echo "proxy_on: cannot read $path" >&2
		return 1
	fi
	if ! raw="$(head -n 1 -- "$path" 2>/dev/null)"; then
		echo "proxy_on: cannot read $path" >&2
		return 1
	fi
	raw="${raw#"${raw%%[![:space:]]*}"}"
	raw="${raw%"${raw##*[![:space:]]}"}"
	if [[ ! "$raw" =~ ^[1-9][0-9]{0,4}$ ]]; then
		echo "proxy_on: invalid port in $path" >&2
		return 1
	fi
	if ((raw < 1 || raw > 65535)); then
		echo "proxy_on: invalid port in $path" >&2
		return 1
	fi
	printf '%s\n' "$raw"
}

proxy_on() {
	local proxy_port proxy_url var
	proxy_port="$(__proxy_resolve)" || return 1
	proxy_url="http://127.0.0.1:${proxy_port}"
	for var in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
		echo "export ${var}=${proxy_url}"
		export "${var}=${proxy_url}"
	done
	echo "export no_proxy=127.0.0.1,localhost"
	export no_proxy="127.0.0.1,localhost"
	echo "export NO_PROXY=127.0.0.1,localhost"
	export NO_PROXY="127.0.0.1,localhost"
	echo -e "\033[32m[√] Proxy enabled\033[0m"
}

proxy_off() {
	local var
	for var in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY no_proxy; do
		echo "unset ${var}"
		unset "$var"
	done
	echo -e "\033[31m[×] Proxy disabled\033[0m"
}

# Optional helper: start a background process if its name is not already running.
start_if_not_running() {
	local process_name="$1"
	shift
	local command="$@"

	if ! pgrep -f "$process_name" >/dev/null; then
		echo "Starting $process_name..."
		eval "$command" &
		sleep 2
		echo "$process_name has been started successfully"
	else
		echo "$process_name is already running"
	fi
}

get_project_root() {
	local dir
	local git_root

	git_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
	if [[ -n "$git_root" && -f "$git_root/.project-root" ]]; then
		echo "$git_root"
		return 0
	fi

	dir="$(pwd)"
	while [[ "$dir" != "/" ]]; do
		if [[ -f "$dir/.project-root" ]]; then
			echo "$dir"
			return 0
		fi
		dir="$(dirname "$dir")"
	done

	echo "Error: Could not find project root (missing .project-root marker)" >&2
	return 1
}

# Optional helper: print an in-place progress bar (current, total).
progress() {
	local current=$1
	local total=$2
	local width=50
	local percent=$((current * 100 / total))
	local filled=$((percent * width / 100))
	local bar
	local space
	bar=$(printf "%${filled}s" | tr ' ' '#')
	space=$(printf "%$((width - filled))s")
	printf "\r[${bar}${space}] %3d%%" "$percent"
}

# Clone or update a git repository (repo_url, target_dir, [branch]).
update_repo() {
	local repo_url=$1
	local target_dir=$2
	local branch=$3
	local stashed=0
	local current_branch
	local current_remote
	local normalized_current_remote
	local normalized_repo_url

	if [[ ! -d "$target_dir" ]]; then
		echo "Cloning repository from $repo_url to $target_dir..."
		git clone "$repo_url" "$target_dir"
	else
		echo "Repository $target_dir already exists, skipping clone."
	fi

	pushd "$target_dir" >/dev/null || {
		echo "Failed to enter $target_dir" >&2
		return 1
	}

	if current_remote="$(git remote get-url origin 2>/dev/null)"; then
		normalized_current_remote="${current_remote%.git}"
		normalized_repo_url="${repo_url%.git}"
		if [[ "$normalized_current_remote" != "$normalized_repo_url" ]]; then
			echo "Updating origin URL from $current_remote to $repo_url..."
			git remote set-url origin "$repo_url"
		fi
	else
		echo "Adding origin remote with URL $repo_url..."
		git remote add origin "$repo_url"
	fi

	if [[ -n "$(git status --porcelain)" ]]; then
		git stash push -u -m "prepare.sh auto-stash"
		stashed=1
	fi

	if current_branch="$(git symbolic-ref --short HEAD 2>/dev/null)"; then
		if [[ -n "$branch" && "$branch" != "$current_branch" ]]; then
			echo "Switching from branch $current_branch to $branch..."
			git switch "$branch"
		else
			branch="$current_branch"
		fi
		echo "Pulling latest changes from branch $branch..."
		git -c pull.rebase=false pull origin "$branch"
	else
		echo "Warning: detached HEAD in $target_dir; pulling without branch switch" >&2
		git -c pull.rebase=false pull
	fi

	if [[ "$stashed" -eq 1 ]]; then
		git stash pop || echo "Warning: could not restore stashed changes in $target_dir" >&2
	fi

	popd >/dev/null
}

# Opt-in proxy on source: export MC_ENABLE_PROXY=1 before sourcing (e.g. resume/main.sh).
if [[ "${MC_ENABLE_PROXY:-}" == "1" ]]; then
	proxy_on
fi
