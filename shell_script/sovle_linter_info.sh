#!/bin/env bash

set -e

max_line_length=120


for dir in src template; do
    autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r $dir
    isort $dir
    black --line-length $max_line_length $dir
done