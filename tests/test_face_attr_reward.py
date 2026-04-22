import json

from dumbledore.gt_schema import DeepFaceGTV1
from rewards.face_attr_reward import compute_score, _parse_json_object


def _gt_vec(emb_offset: float = 0.0) -> str:
    fn = [0.0 + (i + emb_offset) * 0.01 for i in range(512)]
    return DeepFaceGTV1(
        image_path="/tmp/f.jpg",
        facenet512=fn,
        analyze={"age": 30, "gender": "Woman", "emotion": "happy", "race": "asian"},
    ).to_ground_truth_string()


def test_parse_json_with_fence() -> None:
    t = 'Here is:\n```json\n{"a": 1, "b": 2}\n```'
    d = _parse_json_object(t)
    assert d == {"a": 1, "b": 2}


def test_compute_score_perfect() -> None:
    gt = _gt_vec(0.0)
    o = json.loads(gt)
    pred = json.dumps(o, sort_keys=True)
    r = compute_score("ds1", pred, gt)
    assert r >= 0.99


def test_compute_score_wrong_attr() -> None:
    gt = _gt_vec(0.0)
    o = json.loads(gt)
    o["analyze"]["age"] = 99
    pred = json.dumps(o)
    r = compute_score("ds1", pred, gt)
    assert 0.0 < r < 0.99


def test_compute_score_malformed_solution() -> None:
    gt = _gt_vec()
    r = compute_score("ds1", "not json at all", gt)
    assert r == 0.0


def test_schema_mismatch() -> None:
    gt = _gt_vec()
    o = json.loads(gt)
    o["schema_version"] = "0.0"
    pred = json.dumps(o)
    r = compute_score("ds1", pred, gt)
    assert r < 0.99


def test_no_analyze_only_schema_and_emb() -> None:
    fn = [0.01 * i for i in range(512)]
    gt = DeepFaceGTV1(
        image_path="/tmp/x.jpg",
        facenet512=fn,
        analyze={},
    ).to_ground_truth_string()
    o = json.loads(gt)
    pred = json.dumps(o, sort_keys=True)
    r = compute_score("ds1", pred, gt)
    assert r >= 0.99
