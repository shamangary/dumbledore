from dumbledore.face_attr_domains import (
    DEEPFACE_DOMINANT_EMOTIONS,
    DEEPFACE_DOMINANT_GENDERS,
    DEEPFACE_DOMINANT_RACES,
    user_prompt_line_for_ground_truth_key,
)
from dumbledore.pipeline_config import GroundTruthOutputConfig
from dumbledore.gt_schema import build_user_prompt


def test_tuples_match_deepface_families() -> None:
    assert "latino hispanic" in DEEPFACE_DOMINANT_RACES
    assert len(DEEPFACE_DOMINANT_EMOTIONS) == 7
    assert DEEPFACE_DOMINANT_GENDERS == ("Man", "Woman")


def test_build_user_includes_options() -> None:
    g = GroundTruthOutputConfig(age=True, gender=True, emotion=True, race=True, bbox=True, is_real=True)
    u = build_user_prompt(image_id="img_x", output=g)
    assert "Man" in u and "Woman" in u
    assert "neutral" in u and "disgust" in u
    assert "latino hispanic" in u
    assert "0–120" in u or "0-120" in u or "120" in u


def test_user_prompt_line_keys() -> None:
    assert "non-negative" in user_prompt_line_for_ground_truth_key("bbox")
    assert "Man" in user_prompt_line_for_ground_truth_key("gender")
