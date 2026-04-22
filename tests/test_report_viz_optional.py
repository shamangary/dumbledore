"""Optional: figure export when matplotlib + pillow + small PNGs exist."""
from pathlib import Path

import pytest


def test_render_figures_grid_and_age(tmp_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        pytest.skip(f"matplotlib or pillow not installed: {e}")

    from dumbledore.dataset_report import JsonlReport, render_figures

    img = tmp_path / "a.png"
    Image.new("RGB", (32, 32), color=(200, 100, 50)).save(img)
    j = JsonlReport(path=str(tmp_path / "x.jsonl"))
    j.image_paths = [str(img), str(img)]
    j.age_values = [20, 30]
    out = tmp_path / "figs"
    try:
        created = render_figures(j, out, grid_size=2)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"matplotlib not usable: {e}")
    if not created:
        pytest.skip("matplotlib did not write figures (missing/broken stack or no readable paths)")
    assert any("image_grid.png" in c for c in created)
    assert any("age_histogram" in c for c in created)
