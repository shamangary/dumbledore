"""
Integration: JSONL (same shape as extract output) -> build_verl_parquet -> read Parquet columns.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_build_parquet_from_jsonl_roundtrip(tmp_path: Path) -> None:
    try:
        import pandas as pd  # noqa: F401
    except Exception as e:  # noqa: BLE001 — ImportError, binary mismatch, etc.
        pytest.skip(f"pandas not usable: {e}")

    (tmp_path / "i0.png").write_bytes(b"")
    img_a = str(tmp_path / "i0.png")
    gt = json.dumps(
        {
            "0": {
                "bbox": [0, 0, 10, 10],
                "age": 20,
                "gender": "X",
            }
        }
    )
    jf = tmp_path / "g.jsonl"
    jf.write_text(
        json.dumps({"image_path": img_a, "ground_truth": gt, "ok": True}) + "\n"
    )
    outd = tmp_path / "verl"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "dumbledore.cli.build_verl_parquet",
            "--jsonl",
            str(jf),
            "--out-dir",
            str(outd),
            "--train-ratio",
            "1",
            "--val-ratio",
            "0",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    tr = outd / "train.parquet"
    assert tr.is_file()
    import pandas as pd

    df = pd.read_parquet(tr, engine="pyarrow")
    assert len(df) == 1
    assert "prompt" in df.columns
    ex = json.loads(df.iloc[0]["extra_info"])
    assert ex.get("whole_image") is True
    assert "verl_rl" in ex
    assert ex["verl_rl"]["label"]["parquet_column"] == "ground_truth"
    assert "ground_truth" in df.columns
    assert "The file path" not in (df.iloc[0]["prompt"] or "")
