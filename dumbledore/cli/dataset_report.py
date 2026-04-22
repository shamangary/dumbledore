#!/usr/bin/env python3
"""
One-shot stats + optional visualization for JSONL (after extract) and/or Parquet (after build_parquet).

Examples:
  ./scripts/run_stage3_report.sh
  python -m dumbledore.cli.dataset_report --jsonl data/gt.jsonl --parquet-dir data/verl --out reports/run1
  python -m dumbledore.cli.dataset_report --config configs/pipeline.yaml --out reports/run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dumbledore.dataset_report import print_text_summary, render_figures, run_report, write_json
from dumbledore.paths import REPO_ROOT
from dumbledore.pipeline_config import load_pipeline_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset quality stats and optional PNG figures")
    ap.add_argument("--config", type=Path, default=None, help="Pipeline YAML (data.jsonl, data.parquet_dir)")
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--parquet-dir", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for report.json and figures (default: reports/latest under repo)",
    )
    ap.add_argument("--no-viz", action="store_true", help="Skip image grid and age histogram PNGs")
    ap.add_argument("--grid-size", type=int, default=12, help="Max images in grid")
    args = ap.parse_args()

    jsonl: Path | None = args.jsonl
    pdir: Path | None = args.parquet_dir
    if args.config is not None:
        cfg = load_pipeline_config(args.config)
        jsonl = jsonl or Path(cfg.data.jsonl)
        pdir = pdir or Path(cfg.data.parquet_dir)

    out = args.out or (REPO_ROOT / "reports" / "latest")
    out.mkdir(parents=True, exist_ok=True)

    rep = run_report(jsonl=jsonl, parquet_dir=pdir)

    if rep.jsonl is None and rep.parquet is None:
        print("Nothing to report: pass --jsonl and/or --parquet-dir, or --config with existing paths", file=sys.stderr)
        return 1

    print_text_summary(rep)
    write_json(rep, out / "report.json")
    print(f"Wrote {out / 'report.json'}")

    if not args.no_viz and rep.jsonl is not None:
        figs = render_figures(rep.jsonl, out, grid_size=args.grid_size)
        for p in figs:
            print(f"Wrote {p}")
        if not figs and rep.jsonl.image_paths:
            print("(No figures: install matplotlib and pillow, or no readable image files)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
