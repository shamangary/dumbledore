#!/usr/bin/env python3
"""
Download a small face-image set for local DeepFace processing.

Default `dataset.name: lfw` uses torchvision LFWPeople (funneled faces, public / research use;
see torchvision license and http://vis-www.cs.umass.edu/lfw/).

Usage:
  python scripts/download_face_subset.py --config configs/pipeline.yaml
  python scripts/download_face_subset.py --out-dir data/raw_images --num-images 100 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dumbledore.pipeline_config import load_pipeline_config


def _save_lfw(out_dir: Path, num_images: int, seed: int) -> int:
    """LFW funneled faces via scikit-learn (http://vis-www.cs.umass.edu/lfw/)."""
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError as e:
        print("Install: pip install scikit-learn", file=sys.stderr)
        raise SystemExit(1) from e

    import random

    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    # color=True, small resize to keep file count fast
    d = fetch_lfw_people(
        min_faces_per_person=1,
        color=True,
        resize=0.4,
        download_if_missing=True,
    )
    # d.images: (N, h, w) grayscale OR (N, h, w, 3) if color
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
        dest = out_dir / f"lfw_{i:04d}.png"
        img.save(dest)
    print(
        f"Wrote {n_avail} images under {out_dir} (LFW via scikit-learn; see dataset license and cite in publications)"
    )
    return 0 if n_avail else 1


def main() -> int:
    ap = argparse.ArgumentParser()
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

    if dname in ("lfw", "lfw_people", "default"):
        return _save_lfw(out, nimg, seed)
    print(f"Unsupported dataset.name={dname!r} in config; use 'lfw' or implement in download_face_subset.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
