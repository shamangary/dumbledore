#!/usr/bin/env bash
# Re-exec with bash if invoked as `sh script.sh` (dash does not support `pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Stage 4: JSONL → verl Parquet (train/val/test). Optional: --max-rows N
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pipeline_env.sh
source "${SCRIPT_DIR}/pipeline_env.sh"
exec python -m dumbledore.cli.build_verl_parquet --config "$PIPELINE_CONFIG" "$@"
