#!/usr/bin/env bash
set -e

# Execute prepare script
source shell_script/prepare.sh

# Count how many cards are in CUDA_VISIBLE_DEVICES
n_cards=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# Get script arguments (first and all subsequent)
python_file_with_args="${@:1}"

echo "n_cards: $n_cards"
echo "python_file_with_args: $python_file_with_args"

python -m accelerate.commands.launch \
    --main_process_port "$(shuf -i 10000-60000 -n 1)" \
    --config_file "configure/accelerate/acc_cfg_${n_cards}cards.yaml" \
    $python_file_with_args
