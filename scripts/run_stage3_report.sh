#!/usr/bin/env bash
# Re-exec with bash if invoked as `sh script.sh` (dash does not support `pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Stage 3: dataset report on JSONL (run before Parquet; re-run after stage 4 to include split stats). Default --out: reports/latest or $REPORT_DIR
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pipeline_env.sh
source "${SCRIPT_DIR}/pipeline_env.sh"
REPORT_DIR="${REPORT_DIR:-$REPO_ROOT/reports/latest}"
has_out=0
for a in "$@"; do
  if [[ "$a" == --out ]]; then
    has_out=1
    break
  fi
done
if [[ $has_out -eq 0 ]]; then
  exec python -m dumbledore.cli.dataset_report --config "$PIPELINE_CONFIG" --out "$REPORT_DIR" "$@"
else
  exec python -m dumbledore.cli.dataset_report --config "$PIPELINE_CONFIG" "$@"
fi
