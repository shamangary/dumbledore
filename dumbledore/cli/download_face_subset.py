#!/usr/bin/env python3
"""
Download a small face-image set for local DeepFace processing.

- ``dataset.name: lfw`` — LFW funneled faces via **scikit-learn**'s `fetch_lfw_people`
  (see [LFW](http://vis-www.cs.umass.edu/lfw/) and scikit-learn license; cite in publications).
- ``dataset.name: wider_face`` (or ``wider``) — a **random subset** of WIDER FACE from the Hugging
  Face Hub, loaded with **streaming** (no full ~1.5GB archive). Uses ``datasets`` 2.x
  (the Hub’s loading script is not loadable in ``datasets`` 3+). WIDER is **CC BY-NC-ND 4.0** (non-commercial).

Usage:
  python -m dumbledore.cli.download_face_subset --config configs/pipeline.yaml
  python -m dumbledore.cli.download_face_subset --out-dir data/raw_images --num-images 100 --seed 42
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
import traceback
from pathlib import Path

from dumbledore.pipeline_config import load_pipeline_config


def _save_lfw(out_dir: Path, num_images: int, seed: int) -> int:
    """LFW funneled faces via scikit-learn (http://vis-www.cs.umass.edu/lfw/)."""
    try:
        from sklearn.datasets import fetch_lfw_people
    except (ImportError, AttributeError, OSError) as e:
        # AttributeError: _ARRAY_API is typical when NumPy 2.x meets SciPy/sklearn
        # built for NumPy 1.x — align with: pip install -U numpy scipy scikit-learn
        print(
            "scikit-learn (or its SciPy/NumPy stack) failed to import.\n"
            "  pip install -U \"numpy>=2.0\" \"scipy>=1.11\" \"scikit-learn>=1.4\"\n"
            f"  ({e!s})",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    import numpy as np
    from PIL import Image

    # Sklearn LFW is low-res (often ~100px short side at resize=1.0). DeepFace `analyze` on
    # tens-of-px images yields near-identical pseudo-GT across different faces; scale up for the teacher.
    _MIN_SHORT = 160

    def _ensure_min_image_size(pil: Image.Image) -> Image.Image:
        w, h = pil.size
        m = min(w, h)
        if m >= _MIN_SHORT:
            return pil
        s = _MIN_SHORT / m
        return pil.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.Resampling.BICUBIC)

    out_dir.mkdir(parents=True, exist_ok=True)
    # resize is a *downscale factor* on sklearn's LFW output; 0.4 → ~tens-of-px wide images,
    # which breaks DeepFace analyze (nearly identical age/emotion/gender across files).
    d = fetch_lfw_people(
        min_faces_per_person=1,
        color=True,
        resize=1.0,
        download_if_missing=True,
    )
    arr = d.images
    n_avail = min(arr.shape[0], num_images)
    idxs = list(range(arr.shape[0]))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n_avail]
    for i, j in enumerate(idxs):
        im = arr[j]
        if im.ndim == 2:
            img = Image.fromarray((im * 255).astype(np.uint8), mode="L").convert("RGB")
        else:
            img = Image.fromarray((im * 255).astype(np.uint8), mode="RGB")
        img = _ensure_min_image_size(img)
        img.save(out_dir / f"lfw_{i:04d}.png")
    print(
        f"Wrote {n_avail} images under {out_dir} (LFW via scikit-learn; see dataset license and cite in publications)"
    )
    return 0 if n_avail else 1


def _wider_split_for_hub(split: str) -> str:
    s = (split or "train").lower().strip()
    if s in ("val", "validation", "dev"):
        return "validation"
    if s in ("test",):
        return "test"
    if s in ("train", "training", "tr"):
        return "train"
    return "train"


def _datasets_major_version() -> int | None:
    try:
        import datasets  # type: ignore[import-not-found]

        return int(datasets.__version__.split(".", 1)[0])
    except Exception:  # noqa: BLE001
        return None


def _save_wider_face(
    out_dir: Path,
    num_images: int,
    seed: int,
    split: str,
    hf_id: str,
) -> int:
    """WIDER FACE: stream from Hub, shuffle, take `num_images` (subset only)."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        print('Install: pip install -e ".[wider]"\n' f"  ({e!s})", file=sys.stderr)
        raise SystemExit(1) from e
    major = _datasets_major_version()
    if major is not None and major >= 3:
        print(
            "WIDER on the Hub uses a Python loading script; that requires `datasets` 2.x.\n"
            "  pip install \"datasets>=2.16,<3\"\n"
            "  (HuggingFace dropped dataset scripts in `datasets` 3.0+.)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from PIL import Image  # local import; pillow is a core dep

    sp = _wider_split_for_hub(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        stream = load_dataset(
            hf_id,
            split=sp,
            streaming=True,
            trust_remote_code=True,
        )
    except (RuntimeError, OSError) as e:
        err = f"{e!s}"
        if "no longer supported" in err or "not supported" in err.lower() or "Dataset scripts" in err:
            print(
                "The Hub WIDER FACE builder failed (often means `datasets` 3+).\n"
                "  pip install \"datasets>=2.16,<3\"",
                file=sys.stderr,
            )
        raise
    buf = max(1_000, min(20_000, num_images * 100))
    shuf = stream.shuffle(seed=seed, buffer_size=buf)
    n_written = 0
    for i, row in enumerate(itertools.islice(shuf, num_images)):
        img = row.get("image")
        if img is None:
            continue
        if isinstance(img, Image.Image):
            pil = img
        else:
            try:
                import numpy as np

                arr = np.asarray(img)
                pil = Image.fromarray(arr)
            except Exception:  # noqa: BLE001
                continue
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil.save(out_dir / f"wider_{i:04d}.png")
        n_written += 1
    if n_written:
        print(
            f"Wrote {n_written} images under {out_dir} (WIDER FACE subset; streaming; split={sp}; "
            f"CC BY-NC-ND 4.0; cite the WIDER FACE paper and comply with the license.)"
        )
    else:
        print("No WIDER FACE images were written.", file=sys.stderr)
    return 0 if n_written else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download a small face image subset (LFW or WIDER FACE) into a directory of PNGs.",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--num-images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_pipeline_config(args.config) if args.config else None
    out = args.out_dir or (Path(cfg.data.raw_dir) if cfg else Path("data/raw_images"))
    nimg = args.num_images if args.num_images is not None else (cfg.data.num_images if cfg else 100)
    seed = args.seed if args.seed is not None else (cfg.data.seed if cfg else 42)
    dname = (cfg.dataset.name if cfg else "lfw").lower() if cfg else "lfw"
    dsplit = (cfg.dataset.split if cfg else "train")
    dhf = (cfg.dataset.hf_id if cfg else "CUHK-CSE/wider_face")

    if dname in ("wider", "wider_face"):
        try:
            return _save_wider_face(out, nimg, seed, dsplit, dhf)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(
                f"WIDER FACE download failed: {e!s}\n"
                f'  pip install -e ".[wider]"  # needs datasets 2.x for Hub loading script\n',
                file=sys.stderr,
            )
            traceback.print_exc(limit=2, file=sys.stderr)
            return 1
    if dname not in ("lfw", "lfw_people", "default"):
        print(
            f"Unsupported dataset.name={dname!r} in config; use 'lfw' or 'wider_face' (or implement in download_face_subset.py).",
            file=sys.stderr,
        )
        return 1
    try:
        return _save_lfw(out, nimg, seed)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — surface import/binary mismatch from sklearn, scipy, numpy
        print(
            f"LFW download via scikit-learn failed: {e!s}\n"
            f"  If you see NumPy/SciPy '_ARRAY_API' or binary errors, align wheels:  pip install -U numpy scipy scikit-learn\n"
            f"  Or use your own image folder:  run_stage2_extract with --image-dir /path/to/faces (skip stage 1).\n"
            f"Traceback (for maintainers):",
            file=sys.stderr,
        )
        traceback.print_exc(limit=2, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
