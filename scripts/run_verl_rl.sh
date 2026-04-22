#!/usr/bin/env bash
# Read configs/pipeline.yaml (or CONFIG env) and print how to run verl with the custom reward.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Resolves to configs/pipeline.yaml if present, else configs/pipeline.example.yaml (see verl_config_helper.py)
export CONFIG
python "${REPO_ROOT}/scripts/verl_config_helper.py" ${CONFIG:+--config "$CONFIG"} "$@"
