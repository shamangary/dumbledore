"""
Statistics and optional visualizations for JSONL (extract output) and Parquet (verl) stages.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dumbledore.gt_inspect import summarize_ground_truth_string


@dataclass
class JsonlReport:
    path: str
    lines_nonempty: int = 0
    records: int = 0
    top_level_errors: int = 0
    top_level_error_samples: list[str] = field(default_factory=list)
    ground_truth_parse_errors: int = 0
    gt_error_samples: list[str] = field(default_factory=list)
    with_empty_analyze: int = 0
    unique_image_paths: int = 0
    duplicate_image_paths: int = 0
    analyze_key_counts: dict[str, int] = field(default_factory=dict)
    age_values: list[int] = field(default_factory=list)
    gender_counts: dict[str, int] = field(default_factory=dict)
    emotion_counts: dict[str, int] = field(default_factory=dict)
    race_counts: dict[str, int] = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)


@dataclass
class ParquetFileReport:
    name: str
    path: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    whole_image_true: int = 0
    extra_info_parse_errors: int = 0


@dataclass
class ParquetDirReport:
    directory: str
    files: list[ParquetFileReport] = field(default_factory=list)
    total_rows: int = 0


@dataclass
class DatasetReport:
    jsonl: JsonlReport | None = None
    parquet: ParquetDirReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.jsonl is not None:
            d["jsonl"] = asdict(self.jsonl)
        if self.parquet is not None:
            d["parquet"] = {
                "directory": self.parquet.directory,
                "total_rows": self.parquet.total_rows,
                "files": [asdict(f) for f in self.parquet.files],
            }
        return d


def _count_keys(counter: dict[str, int], keys: list[str]) -> None:
    for k in keys:
        counter[k] = counter.get(k, 0) + 1


def analyze_jsonl(path: Path) -> JsonlReport:
    rep = JsonlReport(path=str(path.resolve()))
    if not path.is_file():
        return rep

    path_seen: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rep.lines_nonempty += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                rep.top_level_errors += 1
                if len(rep.top_level_error_samples) < 5:
                    rep.top_level_error_samples.append(f"line {i}: {e!s}")
                continue
            if not isinstance(o, dict):
                rep.top_level_errors += 1
                continue
            rep.records += 1
            ip = o.get("image_path")
            if isinstance(ip, str):
                rep.image_paths.append(ip)
                path_seen[ip] = path_seen.get(ip, 0) + 1
            gts = o.get("ground_truth")
            if not isinstance(gts, str):
                rep.ground_truth_parse_errors += 1
                if len(rep.gt_error_samples) < 5:
                    rep.gt_error_samples.append("missing or non-string ground_truth")
                continue
            sm = summarize_ground_truth_string(gts)
            if not sm.ok:
                rep.ground_truth_parse_errors += 1
                if len(rep.gt_error_samples) < 5:
                    rep.gt_error_samples.append(sm.error or "parse error")
                continue
            if not sm.analyze_keys:
                rep.with_empty_analyze += 1
            _count_keys(rep.analyze_key_counts, sm.analyze_keys)
            if sm.ages:
                rep.age_values.extend(sm.ages)
            elif sm.age is not None:
                rep.age_values.append(sm.age)
            for g in sm.genders or ([sm.gender] if sm.gender else []):
                rep.gender_counts[g] = rep.gender_counts.get(g, 0) + 1
            for e in sm.emotions or ([sm.emotion] if sm.emotion else []):
                rep.emotion_counts[e] = rep.emotion_counts.get(e, 0) + 1
            for r_ in sm.races or ([sm.race] if sm.race else []):
                rep.race_counts[r_] = rep.race_counts.get(r_, 0) + 1

    rep.unique_image_paths = len(path_seen)
    rep.duplicate_image_paths = sum(c - 1 for c in path_seen.values() if c > 1)
    return rep


def analyze_parquet_dir(directory: Path) -> ParquetDirReport:
    import pandas as pd

    out = ParquetDirReport(directory=str(directory.resolve()))
    if not directory.is_dir():
        return out

    for name in ("train", "val", "test"):
        p = directory / f"{name}.parquet"
        if not p.is_file():
            continue
        pr = ParquetFileReport(name=name, path=str(p))
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:  # noqa: BLE001
            continue
        pr.columns = list(df.columns)
        pr.rows = len(df)
        out.total_rows += pr.rows
        if "extra_info" in df.columns:
            for s in df["extra_info"].astype(str):
                try:
                    ex = json.loads(s)
                except json.JSONDecodeError:
                    pr.extra_info_parse_errors += 1
                    continue
                if ex.get("whole_image") is True:
                    pr.whole_image_true += 1
        out.files.append(pr)
    return out


def write_json(report: DatasetReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


def render_figures(
    jsonl_report: JsonlReport,
    out_dir: Path,
    *,
    grid_size: int = 12,
    max_age_bins: int = 20,
) -> list[str]:
    """
    Write PNGs: `image_grid.png` (if PIL can open paths), `age_hist.png` (if ages present).
    Returns list of created file paths (may be empty if viz deps missing or no data).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # ImportError, NumPy/Matplotlib binary mismatch (e.g. AttributeError: _ARRAY_API), etc.
        return []

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    paths = [p for p in jsonl_report.image_paths[:grid_size] if Path(p).is_file()]

    if paths:
        n = min(len(paths), grid_size)
        cols = min(4, max(1, n))
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        flat = axes.flatten()
        for i, ax in enumerate(flat):
            ax.axis("off")
            if i < n:
                try:
                    im = Image.open(paths[i])
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    ax.imshow(im)
                except OSError:
                    ax.text(0.5, 0.5, "unreadable", ha="center", va="center", transform=ax.transAxes)
        p_out = out_dir / "image_grid.png"
        fig.suptitle("Sample images (first N readable paths from JSONL)")
        fig.tight_layout()
        fig.savefig(p_out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(str(p_out))

    if jsonl_report.age_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(jsonl_report.age_values, bins=min(max_age_bins, max(5, len(set(jsonl_report.age_values)))))
        ax.set_xlabel("age (from GT analyze)")
        ax.set_ylabel("count")
        ax.set_title("Age distribution in ground_truth")
        p_out = out_dir / "age_histogram.png"
        fig.tight_layout()
        fig.savefig(p_out, dpi=120)
        plt.close(fig)
        created.append(str(p_out))

    return created


def run_report(
    *,
    jsonl: Path | None = None,
    parquet_dir: Path | None = None,
) -> DatasetReport:
    r = DatasetReport()
    if jsonl is not None:
        r.jsonl = analyze_jsonl(jsonl)
    if parquet_dir is not None:
        r.parquet = analyze_parquet_dir(parquet_dir)
    return r


def print_text_summary(report: DatasetReport) -> None:
    if report.jsonl:
        j = report.jsonl
        print(f"JSONL: {j.path}")
        print(f"  lines (non-empty): {j.lines_nonempty}  records: {j.records}")
        if j.top_level_errors:
            print(f"  top-level JSON errors: {j.top_level_errors}")
        if j.ground_truth_parse_errors:
            print(f"  ground_truth parse errors: {j.ground_truth_parse_errors}")
        print(f"  empty attribute keys: {j.with_empty_analyze}")
        print(f"  unique image_path: {j.unique_image_paths}  duplicate paths: {j.duplicate_image_paths}")
        if j.analyze_key_counts:
            print(f"  analyze keys: {j.analyze_key_counts}")
    if report.parquet and report.parquet.files:
        p = report.parquet
        print(f"Parquet dir: {p.directory}  total rows: {p.total_rows}")
        for f in p.files:
            print(f"  {f.name}: {f.rows} rows  columns={f.columns}  whole_image@true: {f.whole_image_true}")
