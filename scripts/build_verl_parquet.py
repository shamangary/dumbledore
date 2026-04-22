#!/usr/bin/env python3
"""
Build verl-style Parquet from `extract_deepface_gt.py` JSONL: train/val/test splits.

Usage:
  python scripts/build_verl_parquet.py --config configs/pipeline.yaml
  python scripts/build_verl_parquet.py --jsonl data/gt.jsonl --out-dir data/verl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dumbledore.gt_schema import SYSTEM_INSTRUCTION, build_user_prompt, DEFAULT_ANALYZE_ACTIONS
from dumbledore.pipeline_config import PipelineConfig, load_pipeline_config, SUPPORTED_ANALYZE_ATTRIBUTES


def _image_id_for_path(p: str) -> str:
    h = hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]
    return f"img_{h}"


def _full_prompt_text(
    image_id: str,
    image_path: str,
    analyze_keys: tuple[str, ...] | None,
    include_facenet512: bool,
) -> str:
    user = build_user_prompt(
        image_id=image_id,
        image_path=image_path,
        analyze_keys=analyze_keys,
        include_facenet512=include_facenet512,
    )
    return f"{SYSTEM_INSTRUCTION}\n\n{user}"


def _parse_gt_keys(gt_str: str) -> tuple[list[str], bool]:
    """Infer analyze keys and whether facenet expected from one ground_truth JSON string."""
    o = json.loads(gt_str)
    an = o.get("analyze") if isinstance(o.get("analyze"), dict) else {}
    keys = [k for k in SUPPORTED_ANALYZE_ATTRIBUTES if k in an]
    if not keys and an:
        keys = list(an.keys())
    fn = o.get("facenet512")
    has_fn = isinstance(fn, list) and len(fn) == 512
    return keys, has_fn


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

    jsonl = args.jsonl or (Path(cfg.data.jsonl) if cfg else None)
    out_dir = args.out_dir or (Path(cfg.data.parquet_dir) if cfg else None)
    if jsonl is None or out_dir is None:
        print("Need --jsonl and --out-dir, or --config with data.jsonl and data.parquet_dir", file=sys.stderr)
        return 1

    train_ratio = args.train_ratio if args.train_ratio is not None else (cfg.data.train_ratio if cfg else 0.8)
    val_ratio = args.val_ratio if args.val_ratio is not None else (cfg.data.val_ratio if cfg else 0.1)
    seed = args.seed if args.seed is not None else (cfg.data.seed if cfg else 42)
    max_rows = args.max_rows

    analyze_keys_cfg: tuple[str, ...] | None = None
    include_fn_cfg = True
    if cfg is not None:
        analyze_keys_cfg = tuple(cfg.deepface.enabled_actions())
        include_fn_cfg = cfg.deepface.include_facenet512

    if not jsonl.is_file():
        print(f"Not found: {jsonl}", file=sys.stderr)
        return 1
    r = train_ratio + val_ratio
    if r >= 1.0 or train_ratio <= 0 or val_ratio < 0:
        print("Invalid split ratios; train+val must be < 1.0, train>0, val>=0", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ip = o["image_path"]
            gt = o["ground_truth"]
            iid = _image_id_for_path(ip)

            keys_inf, has_fn_inf = _parse_gt_keys(gt)
            keys = list(analyze_keys_cfg) if analyze_keys_cfg is not None else keys_inf
            include_fn = include_fn_cfg if cfg is not None else has_fn_inf

            prompt = _full_prompt_text(iid, ip, tuple(keys) if keys else None, include_fn)
            extra = json.dumps(
                {
                    "image_id": iid,
                    "image_path": ip,
                    "modality": "text_or_vlm",
                    "whole_image": True,
                    "analyze_keys": keys,
                    "include_facenet512": include_fn,
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
