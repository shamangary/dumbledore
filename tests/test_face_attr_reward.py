import json

from rewards.face_attr_reward import compute_score, _parse_json_object


def _gt_multi(age_a: int = 30, age_b: int | None = None) -> str:
    f1 = {"bbox": [0, 0, 10, 10], "age": age_a, "emotion": "happy", "gender": "Woman", "race": "asian"}
    d: dict[str, dict] = {"0": f1}
    if age_b is not None:
        d["1"] = {
            "bbox": [100, 0, 10, 10],
            "age": age_b,
            "emotion": "neutral",
            "gender": "Man",
            "race": "white",
        }
    return json.dumps(d, sort_keys=True)


def test_parse_json_with_fence() -> None:
    t = 'Here is:\n```json\n{"a": 1, "b": 2}\n```'
    d = _parse_json_object(t)
    assert d == {"a": 1, "b": 2}


def test_compute_score_perfect() -> None:
    gt = _gt_multi()
    r = compute_score("ds1", gt, gt)
    assert r >= 0.99


def test_compute_score_wrong_attr() -> None:
    gt = _gt_multi()
    o = json.loads(gt)
    o["0"]["age"] = 90
    r = compute_score("ds1", json.dumps(o), gt)
    assert 0.0 < r < 0.99


def test_malformed_solution() -> None:
    gt = _gt_multi()
    r = compute_score("ds1", "not json at all", gt)
    assert r == 0.0


def test_two_face_partial_mismatch() -> None:
    gt = _gt_multi(30, 40)
    o = json.loads(gt)
    o["1"]["age"] = 99
    r = compute_score("ds1", json.dumps(o), gt)
    assert 0.0 < r < 1.0


def test_both_empty_objects_perfect() -> None:
    assert compute_score("ds1", "{}", "{}") == 1.0


def test_perfect_match() -> None:
    gt = _gt_multi()
    r = compute_score("ds1", gt, gt)
    assert r >= 0.99
