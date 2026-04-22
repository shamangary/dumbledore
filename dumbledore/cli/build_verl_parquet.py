#!/usr/bin/env python3
"""
Build verl-style Parquet from `extract_deepface_gt` JSONL: train/val/test splits.

Usage:
  ./scripts/run_stage4_parquet.sh
  python -m dumbledore.cli.build_verl_parquet --config configs/pipeline.yaml
  python -m dumbledore.cli.build_verl_parquet --jsonl data/gt.jsonl --out-dir data/verl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from dumbledore.gt_inspect import infer_ground_truth_output_config, parse_gt_keys
from dumbledore.pipeline_config import GroundTruthOutputConfig, PipelineConfig, load_pipeline_config
from dumbledore.prompts import (
    build_training_prompt,
    dataset_prompt_key,
    get_effective_prompt_config,
    image_id_for_path,
    list_prompt_ground_truth_mismatches,
)


def _verl_rl_row_contract() -> dict:
    """How verl / RL should interpret each Parquet row: text request + image + pseudo-GT target."""
    return {
        "inputs": {
            "text": {"parquet_column": "prompt", "role": "user_request_and_json_schema"},
            "image": {
                "source": "absolute path in this JSON at key image_path (same as JSONL `image_path`)",
                "role": "full_frame_vlm_context",
                "whole_image": True,
            },
        },
        "label": {
            "parquet_column": "ground_truth",
            "role": "pseudo_ground_truth_from_deepface",
            "reward": "rewards/face_attr_reward.py:compute_score(solution_str, ground_truth, …)",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build verl Parquet from DeepFace JSONL")
    ap.add_argument("--config", type=Path, default=None, help="Master pipeline YAML")
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--val-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="Use at most N JSONL rows after load")
    ap.add_argument("--data-source", type=str, default="deepface_face_attr")
    ap.add_argument("--ability", type=str, default="face_attr_json")
    args = ap.parse_args()

    cfg: PipelineConfig | None = None
    if args.config is not None:
        cfg = load_pipeline_config(args.config)
        for m in list_prompt_ground_truth_mismatches(cfg):
            print(f"Warning: prompt vs deepface.ground_truth: {m}", file=sys.stderr)

    jsonl = args.jsonl or (Path(cfg.data.jsonl) if cfg else None)
    out_dir = args.out_dir or (Path(cfg.data.parquet_dir) if cfg else None)
    if jsonl is None or out_dir is None:
        print("Need --jsonl and --out-dir, or --config with data.jsonl and data.parquet_dir", file=sys.stderr)
        return 1

    train_ratio = args.train_ratio if args.train_ratio is not None else (cfg.data.train_ratio if cfg else 0.8)
    val_ratio = args.val_ratio if args.val_ratio is not None else (cfg.data.val_ratio if cfg else 0.1)
    seed = args.seed if args.seed is not None else (cfg.data.seed if cfg else 42)
    max_rows = args.max_rows

    out_cfg: GroundTruthOutputConfig | None = None
    include_fn_cfg = True
    if cfg is not None:
        out_cfg = cfg.deepface.ground_truth
        include_fn_cfg = cfg.deepface.include_facenet512
    if not jsonl.is_file():
        print(f"Not found: {jsonl}", file=sys.stderr)
        return 1
    r = train_ratio + val_ratio
    if r > 1.0 or train_ratio <= 0 or val_ratio < 0:
        print("Invalid split ratios; train+val must be <= 1.0, train>0, val>=0", file=sys.stderr)
        return 1

    rows: list[dict] = []
    out_fallback: GroundTruthOutputConfig | None = out_cfg
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ip = o["image_path"]
            gt = o["ground_truth"]
            iid = image_id_for_path(ip)
            if out_fallback is None:
                out_fallback = infer_ground_truth_output_config(gt)
            ocfg = out_cfg or out_fallback
            # Facenet embeddings are not stored in `ground_truth`; this flag is from YAML or defaults to True.
            include_fn = include_fn_cfg if cfg is not None else True

            if cfg is not None:
                pcfg = get_effective_prompt_config(cfg)
                dkey = dataset_prompt_key(cfg.dataset.name)
                prompt = build_training_prompt(iid, ocfg, pcfg, dataset_key=dkey)
            elif isinstance(o.get("prompt"), str) and o["prompt"].strip():
                prompt = o["prompt"]
            else:
                prompt = build_training_prompt(iid, ocfg, None)
            keys = parse_gt_keys(gt)
            extra = json.dumps(
                {
                    "image_id": iid,
                    "image_path": ip,
                    "modality": "text_or_vlm",
                    "whole_image": True,
                    "ground_truth_keys": keys,
                    "include_facenet512": include_fn,
                    "verl_rl": _verl_rl_row_contract(),
                },
                ensure_ascii=False,
            )
            rows.append(
                {
                    "data_source": args.data_source,
                    "prompt": prompt,
                    "ability": args.ability,
                    "ground_truth": gt,
                    "extra_info": extra,
                }
            )

    if max_rows is not None:
        rows = rows[: max(0, max_rows)]

    if not rows:
        print("No rows in JSONL", file=sys.stderr)
        return 1

    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train == 0:
        print(f"Warning: train split empty (n={n}, train_ratio={train_ratio}); writing all to train.", file=sys.stderr)
        n_train = n
        n_val = 0

    random.seed(seed)
    random.shuffle(rows)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        if not part:
            continue
        df = pd.DataFrame(part)
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"Wrote {path} ({len(part)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
