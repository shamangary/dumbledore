#!/usr/bin/env python3
"""
Run DeepFace on a directory of face images; write JSONL with `image_path`, `ground_truth`,
`prompt` (from `pipeline.yaml` `prompt` + `deepface.ground_truth` when using `--config`), and
indexed GT (`{"0": { ... }, "1": { ... }}`; no path or embedding inside the GT string).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

from dumbledore.deepface_ops import extract_ground_truth_string
from dumbledore.gt_schema import DEFAULT_ANALYZE_ACTIONS
from dumbledore.pipeline_config import GroundTruthOutputConfig, PipelineConfig, load_pipeline_config
from dumbledore.prompts import (
    build_training_prompt,
    dataset_prompt_key,
    get_effective_prompt_config,
    image_id_for_path,
    list_prompt_ground_truth_mismatches,
)

logger = logging.getLogger(__name__)


def _list_images(root: Path, extensions: frozenset[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract DeepFace indexed per-face GT to JSONL.")
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
        "--max-faces",
        type=int,
        default=None,
        help="Override pipeline max_faces (cap on faces per image after sort)",
    )
    parser.add_argument(
        "--no-facenet512",
        action="store_true",
        help="Do not run represent(); cannot combine with bbox/is_real/embedding check",
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
    except ImportError as e:
        print(
            "DeepFace (and a TensorFlow stack) is required for stage 2.\n"
            "  pip install -e \".[face]\"\n"
            "or:  pip install deepface tf-keras \"tensorflow>=2.0\"  (match NumPy/CPU–GPU to your system)\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    gtcfg: GroundTruthOutputConfig = (
        cfg.deepface.ground_truth if cfg is not None else GroundTruthOutputConfig()
    )
    if not gtcfg.enabled_list():
        print("Refusing: deepface.ground_truth has no fields enabled in config (use --config with ground_truth: …).", file=sys.stderr)
        return 1

    if args.no_analyze:
        actions: tuple[str, ...] = ()
    elif args.actions is not None:
        actions = tuple(s.strip() for s in args.actions.split(",") if s.strip())
    elif cfg is not None:
        actions = tuple(cfg.deepface.enabled_analyze_actions())
    else:
        actions = DEFAULT_ANALYZE_ACTIONS

    include_fn = cfg.deepface.include_facenet512 if cfg else True
    if args.no_facenet512:
        include_fn = False

    need_represent = bool(include_fn or gtcfg.bbox or gtcfg.is_real)
    if not need_represent and not actions:
        print(
            "Refusing: need Facenet `represent` (set bbox, is_real, and/or include_facenet512) "
            "or at least one analyze action.",
            file=sys.stderr,
        )
        return 1

    det = args.detector_backend or (cfg.deepface.detector_backend if cfg else "opencv")
    model_name = args.model_name or (cfg.deepface.model_name if cfg else "Facenet512")
    max_faces = int(args.max_faces) if args.max_faces is not None else (cfg.deepface.max_faces if cfg else 5)
    if max_faces < 1:
        max_faces = 1

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if cfg is not None:
        for m in list_prompt_ground_truth_mismatches(cfg):
            logger.warning("Prompt vs deepface.ground_truth: %s", m)
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
                gt = extract_ground_truth_string(
                    pstr,
                    DeepFace,
                    output=gtcfg,
                    include_facenet512=include_fn,
                    actions=actions,
                    detector_backend=det,
                    model_name=model_name,
                    max_faces=max_faces,
                )
                iid = image_id_for_path(pstr)
                pcfg = get_effective_prompt_config(cfg) if cfg is not None else None
                dkey = dataset_prompt_key(cfg.dataset.name) if cfg is not None else None
                rec = {
                    "image_path": pstr,
                    "ground_truth": gt,
                    "prompt": build_training_prompt(iid, gtcfg, pcfg, dataset_key=dkey),
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
