#!/usr/bin/env python3
"""
Print shell exports and a suggested verl invocation from `configs/pipeline.yaml`.

verl is not bundled; you must `pip install` or clone a pinned release, then run the
entry point your version documents (PPO, GRPO, etc.).

From the repo: ``python -m dumbledore.cli.verl_config`` (or ``./scripts/run_stage5_verl.sh``).
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from dumbledore.paths import REPO_ROOT
from dumbledore.pipeline_config import load_pipeline_config


def main() -> int:
    ex = REPO_ROOT / "configs" / "pipeline.example.yaml"
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Pipeline YAML; defaults to configs/pipeline.yaml if present, else pipeline.example.yaml",
    )
    ap.add_argument("--print-exports", action="store_true", help="Print export VAR=value for bash eval")
    ap.add_argument("--print-body", action="store_true", help="Print only suggested Hydra override lines (no comments)")
    args = ap.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        p_user = REPO_ROOT / "configs" / "pipeline.yaml"
        cfg_path = p_user if p_user.is_file() else ex
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    args.config = cfg_path
    c = load_pipeline_config(args.config)
    pq = Path(c.data.parquet_dir) / "train.parquet"
    reward = REPO_ROOT / "rewards" / "face_attr_reward.py"
    pqt = str(pq.resolve()) if pq.is_file() or pq.parent.is_dir() else str(pq)
    rewt = str(reward.resolve())

    if args.print_exports:
        print(f'export DUMBLEDORE_HF_MODEL_ID="{c.hf_model_id}"')
        print(f'export DUMBLEDORE_VERL_METHOD="{c.verl.method}"')
        print(f'export DUMBLEDORE_TRAIN_PARQUET="{pqt}"')
        print(f'export DUMBLEDORE_REWARD_PATH="{rewt}"')
        return 0
    if args.print_body:
        print(f"custom_reward_function.path={rewt} \\")
        print(f"custom_reward_function.name=compute_score \\")
        print(f"data.train_files={pqt} \\")
        return 0

    print("=== Pipeline (from YAML) ===")
    print(f"hf_model_id: {c.hf_model_id}")
    print(f"verl.method: {c.verl.method}")
    print(f"train.parquet: {pqt} (build with `python -m dumbledore.cli.build_verl_parquet` first)")
    print()
    print("VLM+RL data row (Dumbledore):")
    print("  • model inputs:  text = column `prompt` (request); image = file at `image_path` inside `extra_info` (full frame).")
    print("  • target for RL: column `ground_truth` (pseudo-GT string from DeepFace); `compute_score` compares rollout text to it.")
    print()
    print("Bash (eval exports):")
    print(
        f'  eval "$(python -m dumbledore.cli.verl_config --config {shlex.quote(str(args.config))} --print-exports)"'
    )
    print()
    print("verl: install a pinned release, then use its example (GRPO / PPO / ...).")
    print("Typical extra Hydra overrides (append to the official example command):")
    print()
    print(f"  +custom_reward_function.path={rewt} \\")
    print("  +custom_reward_function.name=compute_score \\")
    print(f"  +data.train_files={pqt} \\")
    print(f"  +actor_rollout_ref.model.path={c.hf_model_id}  # or your config's model key name")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
