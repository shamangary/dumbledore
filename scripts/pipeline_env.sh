# Shared by run_stage*.sh — source after setting SCRIPT_DIR to the scripts/ directory.
# Sets REPO_ROOT, cd's to repo, PYTHONPATH, PIPELINE_CONFIG (and exports CONFIG as an alias of PIPELINE_CONFIG).
# Override config:   CONFIG=configs/pipeline.yaml ./scripts/run_stage2_extract.sh
# Report output dir: REPORT_DIR=reports/qa ./scripts/run_stage3_report.sh
: "${SCRIPT_DIR:?run_stage scripts must set SCRIPT_DIR to scripts/ before sourcing pipeline_env.sh}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT" || exit 1
# Use repo .venv when present so stage 1 (sklearn) / NumPy / SciPy wheels match the venv, not a broken system conda.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  _VENV_BIN="${REPO_ROOT}/.venv/bin"
  export PATH="${_VENV_BIN}:${PATH}"
  unset _VENV_BIN
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

_resolve_pipeline_config() {
  if [[ -n "${CONFIG:-}" ]]; then
    local c="$CONFIG"
    if [[ ! -f "$c" && -f "${REPO_ROOT}/${c#./}" ]]; then
      c="${REPO_ROOT}/${c#./}"
    fi
    if [[ -f "$c" ]]; then
      echo "$(cd "$(dirname "$c")" && pwd)/$(basename "$c")"
      return 0
    fi
    echo "error: config file not found: $CONFIG" >&2
    return 1
  fi
  if [[ -f "${REPO_ROOT}/configs/pipeline.yaml" ]]; then
    echo "${REPO_ROOT}/configs/pipeline.yaml"
    return 0
  fi
  echo "${REPO_ROOT}/configs/pipeline.example.yaml"
}
PIPELINE_CONFIG="$(_resolve_pipeline_config)" || exit 1
export CONFIG="$PIPELINE_CONFIG"
export PIPELINE_CONFIG
