import json

from dumbledore.deepface_ops import build_one_face_from_deepface, normalize_analyze_dict
from dumbledore.gt_schema import build_indexed_ground_truth_string
from dumbledore.pipeline_config import GroundTruthOutputConfig


def test_normalize_analyze_dict() -> None:
    face = {
        "age": 25,
        "dominant_gender": "Woman",
        "dominant_emotion": "happy",
        "dominant_race": "asian",
    }
    d = normalize_analyze_dict(face, ("age", "gender", "emotion", "race"))
    assert d["age"] == 25
    assert d["gender"] == "Woman"
    assert d["emotion"] == "happy"
    assert d["race"] == "asian"


def test_build_one_face_from_deepface() -> None:
    out = GroundTruthOutputConfig(
        bbox=True, age=True, gender=True, emotion=False, race=False, is_real=False
    )
    rep = [{"facial_area": {"x": 1, "y": 2, "w": 3, "h": 4}, "embedding": [0.0] * 512}]
    analyze_list = [
        {
            "age": 20,
            "dominant_gender": "Man",
        }
    ]
    d = build_one_face_from_deepface(
        rep,
        analyze_list,
        output=out,
        include_facenet512=True,
        actions=("age", "gender", "emotion", "race"),
    )
    s = build_indexed_ground_truth_string([d])
    o = json.loads(s)
    assert o["0"]["bbox"] == [1, 2, 3, 4]
    assert o["0"]["age"] == 20
    assert o["0"]["gender"] == "Man"
