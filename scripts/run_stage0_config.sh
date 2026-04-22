#!/usr/bin/env bash
# Re-exec with bash if invoked as `sh script.sh` (dash does not support `pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
# Stage 0: create configs/pipeline.yaml from the default example if missing (or --force to overwrite).
# Other examples: configs/pipeline.lfw.example.yaml, configs/pipeline.wider.example.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pipeline_env.sh
source "${SCRIPT_DIR}/pipeline_env.sh"
EX="${REPO_ROOT}/configs/pipeline.example.yaml"
OUT="${REPO_ROOT}/configs/pipeline.yaml"
if [[ "${1:-}" == "--force" ]]; then
  cp -f "$EX" "$OUT"
  echo "Wrote $OUT (overwritten from example)"
elif [[ -f "$OUT" ]]; then
  echo "Config already exists: $OUT"
  echo "Edit it, or run:  $0 --force  (replace with example)"
else
  cp "$EX" "$OUT"
  echo "Created $OUT — edit hf_model_id, deepface, data.*, then run stage 1 (or 2 if you have images). For LFW/WIDER-specific examples, see configs/pipeline.lfw / pipeline.wider.example.yaml"
fi
echo "Next: ./scripts/run_stage1_images.sh   (or skip to ./scripts/run_stage2_extract.sh with your own --image-dir)"
