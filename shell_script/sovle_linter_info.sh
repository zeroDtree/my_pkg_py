#!/bin/env bash

set -e

ruff check --unsafe-fixes --fix
ruff format