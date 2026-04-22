#!/usr/bin/env bash
# Re-exec with bash if invoked as `sh script.sh` (dash does not support `pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Stage 1: download a subset of a Hub dataset (e.g. LFW) into data.raw_dir. Skip if you already have images.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pipeline_env.sh
source "${SCRIPT_DIR}/pipeline_env.sh"
exec python -m dumbledore.cli.download_face_subset --config "$PIPELINE_CONFIG" "$@"
