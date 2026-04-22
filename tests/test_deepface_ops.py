from dumbledore.deepface_ops import build_gt_from_deepface, normalize_analyze_dict


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


def test_build_gt_from_deepface() -> None:
    emb = [0.0] * 512
    analyze_list = [
        {
            "age": 20,
            "dominant_gender": "Man",
        }
    ]
    gt = build_gt_from_deepface(
        "/tmp/x.jpg", emb, analyze_list, actions=("age", "gender", "emotion", "race")
    )
    assert len(gt.facenet512) == 512
    assert "age" in gt.analyze
