"""Shipped example YAML: `deepface.ground_truth` and custom `prompt.user` stay in sync."""
from __future__ import annotations

from pathlib import Path

import pytest

from dumbledore.pipeline_config import (
    DeepFaceConfig,
    GroundTruthOutputConfig,
    PipelineConfig,
    PromptConfig,
    load_pipeline_config,
)
from dumbledore.prompts import list_prompt_ground_truth_mismatches

REPO = Path(__file__).resolve().parents[1]
_CONFIGS = REPO / "configs"


@pytest.mark.parametrize(
    "name",
    [
        "pipeline.example.yaml",
        "pipeline.lfw.example.yaml",
        "pipeline.wider.example.yaml",
    ],
)
def test_example_yaml_prompt_matches_ground_truth(name: str) -> None:
    c = load_pipeline_config(_CONFIGS / name)
    problems = list_prompt_ground_truth_mismatches(c)
    assert not problems, problems


def test_detects_json_field_when_ground_truth_off() -> None:
    c = PipelineConfig(
        deepface=DeepFaceConfig(
            ground_truth=GroundTruthOutputConfig(
                bbox=False,
                age=True,
                gender=True,
                emotion=True,
                race=True,
                is_real=False,
            ),
        ),
    )
    c.prompt = PromptConfig(
        user='Return JSON with "bbox": [1,2,3,4]  # oops, bbox in template but off in config'
    )
    p = list_prompt_ground_truth_mismatches(c)
    assert any("bbox" in x and "false" in x for x in p)
