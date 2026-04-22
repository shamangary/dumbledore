#!/usr/bin/env python3
"""
Run DeepFace on a directory of face images; write JSONL with `ground_truth` (DeepFaceGTV1).

Usage:
  python scripts/extract_deepface_gt.py --config configs/pipeline.yaml
  python scripts/extract_deepface_gt.py --image-dir /path --out data/gt.jsonl --actions age,gender
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dumbledore.deepface_ops import build_gt_from_deepface
from dumbledore.gt_schema import DEFAULT_ANALYZE_ACTIONS
from dumbledore.pipeline_config import PipelineConfig, load_pipeline_config

logger = logging.getLogger(__name__)


def _list_images(root: Path, extensions: frozenset[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DeepFace GT (Facenet512 + analyze) to JSONL.")
    parser.add_argument("--config", type=Path, default=None, help="Master pipeline YAML (overrides below if set)")
    parser.add_argument("--image-dir", type=Path, default=None, help="Root directory of images")
    parser.add_argument("--out", type=Path, default=None, help="Output JSONL path")
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="Comma-separated DeepFace analyze actions (overrides YAML if set)",
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default=None,
        help="Face detector (e.g. opencv, mtcnn)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="DeepFace.represent model name",
    )
    parser.add_argument(
        "--no-facenet512",
        action="store_true",
        help="Omit Facenet512 from GT (override: store empty embedding list)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Only run represent(); skip analyze (overrides attribute list)",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Process at most N image files (sorted order)")
    parser.add_argument("--log-failures", type=Path, default=None, help="Append failed paths to this file")
    args = parser.parse_args()

    cfg: PipelineConfig | None = None
    if args.config is not None:
        cfg = load_pipeline_config(args.config)

    image_dir = args.image_dir or (Path(cfg.data.raw_dir) if cfg else None)
    out_path = args.out or (Path(cfg.data.jsonl) if cfg else None)
    if image_dir is None or out_path is None:
        print("Need --image-dir and --out, or --config with data.raw_dir and data.jsonl", file=sys.stderr)
        return 1

    try:
        from deepface import DeepFace
    except ImportError:
        print("Install deepface: pip install deepface", file=sys.stderr)
        return 1

    if args.no_analyze:
        actions: tuple[str, ...] = ()
    elif args.actions is not None:
        actions = tuple(s.strip() for s in args.actions.split(",") if s.strip())
    elif cfg is not None:
        actions = tuple(cfg.deepface.enabled_actions())
    else:
        actions = DEFAULT_ANALYZE_ACTIONS

    include_fn = cfg.deepface.include_facenet512 if cfg else True
    if args.no_facenet512:
        include_fn = False

    if not include_fn and not actions:
        print("Refusing: include_facenet512 is false and no analyze actions enabled.", file=sys.stderr)
        return 1

    det = args.detector_backend or (cfg.deepface.detector_backend if cfg else "opencv")
    model_name = args.model_name or (cfg.deepface.model_name if cfg else "Facenet512")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    exts = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    if not image_dir.is_dir():
        logger.error("Not a directory: %s", image_dir)
        return 1

    images = _list_images(image_dir, exts)
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    elif cfg is not None:
        images = images[: max(0, cfg.data.num_images)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_fail = 0

    with ExitStack() as stack:
        fout = stack.enter_context(open(out_path, "w", encoding="utf-8"))
        fail_f = None
        if args.log_failures:
            args.log_failures.parent.mkdir(parents=True, exist_ok=True)
            fail_f = stack.enter_context(open(args.log_failures, "a", encoding="utf-8"))

        for img_path in images:
            pstr = str(img_path.resolve())
            try:
                emb = None
                if include_fn:
                    rep = DeepFace.represent(
                        img_path=pstr,
                        model_name=model_name,
                        enforce_detection=True,
                        detector_backend=det,
                    )
                    if isinstance(rep, list) and rep:
                        emb = rep[0].get("embedding")
                    elif isinstance(rep, dict):
                        emb = rep.get("embedding")
                    if not isinstance(emb, list) or len(emb) != 512:
                        raise ValueError("embedding not length 512")

                an_out = None
                if actions:
                    an_out = DeepFace.analyze(
                        img_path=pstr,
                        actions=list(actions),
                        enforce_detection=True,
                        detector_backend=det,
                    )

                gt = build_gt_from_deepface(
                    pstr,
                    emb,
                    an_out,
                    actions=actions,
                    detector_backend=det,
                    model_name=model_name,
                )
                rec = {
                    "image_path": pstr,
                    "ground_truth": gt.to_ground_truth_string(),
                    "ok": True,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                logger.warning("FAIL %s: %s", pstr, e)
                if fail_f is not None:
                    fail_f.write(f"{pstr}\t{repr(e)}\n")

    logger.info("Wrote %s (%d ok, %d fail)", out_path, n_ok, n_fail)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
