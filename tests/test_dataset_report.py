import json
from pathlib import Path

import pytest

from dumbledore.dataset_report import analyze_jsonl, analyze_parquet_dir, run_report, write_json


def _gt_str(age: int, gender: str) -> str:
    return json.dumps({"0": {"age": age, "gender": gender, "bbox": [0, 0, 1, 1]}})


def test_analyze_jsonl_stats(tmp_path: Path) -> None:
    a = str(tmp_path / "a.png")
    b = str(tmp_path / "b.png")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"y")
    lines = [
        json.dumps({"image_path": a, "ground_truth": _gt_str(30, "Woman")}, ensure_ascii=False),
        json.dumps({"image_path": b, "ground_truth": _gt_str(35, "Woman")}, ensure_ascii=False),
    ]
    jf = tmp_path / "t.jsonl"
    jf.write_text("\n".join(lines) + "\n")

    rep = analyze_jsonl(jf)
    assert rep.records == 2
    assert rep.unique_image_paths == 2
    assert len(rep.age_values) == 2
    assert rep.gender_counts.get("Woman") == 2


def test_run_report_writes_json(tmp_path: Path) -> None:
    jf = tmp_path / "x.jsonl"
    jf.write_text(
        json.dumps(
            {
                "image_path": "/dev/null",
                "ground_truth": json.dumps({}),
            }
        )
        + "\n"
    )
    r = run_report(jsonl=jf, parquet_dir=None)
    out = tmp_path / "out.json"
    write_json(r, out)
    data = json.loads(out.read_text())
    assert "jsonl" in data
    assert data["jsonl"]["records"] == 1


def test_analyze_parquet_minimal(tmp_path: Path) -> None:
    try:
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"pandas not usable: {e}")
    df = pd.DataFrame(
        [
            {
                "data_source": "x",
                "prompt": "p",
                "ability": "a",
                "ground_truth": json.dumps({}),
                "extra_info": json.dumps({"whole_image": True}),
            }
        ]
    )
    pdir = tmp_path / "p"
    pdir.mkdir()
    df.to_parquet(pdir / "train.parquet", index=False, engine="pyarrow")
    rep = analyze_parquet_dir(pdir)
    assert rep.total_rows == 1
    assert rep.files[0].whole_image_true == 1
