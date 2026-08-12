#!/usr/bin/env bash
set -e

proxy_ip="127.0.0.1"
proxy_port="17890"

proxy_on() {
    for var in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
        cmd="export $var=http://${proxy_ip}:${proxy_port}"
        # echo "$cmd"
        eval "$cmd"
    done

    export no_proxy="127.0.0.1,localhost"
    export NO_PROXY="127.0.0.1,localhost"

    echo -e "\033[32m[√] Proxy enabled on ${proxy_ip}:${proxy_port} \033[0m"
}

proxy_off() {
    for var in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY no_proxy; do
        cmd="unset $var"
        # echo "$cmd"
        eval "$cmd"
    done

    echo -e "\033[31m[×] Proxy disabled on ${proxy_ip}:${proxy_port} \033[0m"
}

function start_if_not_running() {
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
    # Start from current directory and go up to find project root
    local current_dir="$(pwd)"

    # Check if current directory is project root
    if [[ -f "$current_dir/main.py" && -f "$current_dir/README.md" && -d "$current_dir/data_utils" ]]; then
        echo "$current_dir"
        return 0
    fi

    # Try git root
    local git_root="$(git rev-parse --show-toplevel 2>/dev/null || echo "")"
    if [[ -n "$git_root" && -f "$git_root/main.py" ]]; then
        echo "$git_root"
        return 0
    fi

    echo "Error: Could not find project root" >&2
    return 1
}

clash_start_cmd="/home/dell/software/clash/clash-linux-amd64-2023.08.17 -f /home/dell/work_dir/zengls/conf/ghelp_clash_config.yaml"

# start_if_not_running clash "$clash_start_cmd"

# Assume proxy_on is a function or script
proxy_on


progress() {
    local current=$1
    local total=$2
    local width=50
    local percent=$((current * 100 / total))
    local filled=$((percent * width / 100))
    local bar=$(printf "%${filled}s" | tr ' ' '#')
    local space=$(printf "%$((width - filled))s")
    printf "\r[${bar}${space}] %3d%%" $percent
}
