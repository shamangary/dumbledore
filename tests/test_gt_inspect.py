import json

from dumbledore.gt_inspect import parse_gt_keys, summarize_ground_truth_string


def test_parse_gt_keys_indexed() -> None:
    s = json.dumps(
        {
            "0": {"age": 40, "gender": "Man", "bbox": [0, 0, 1, 1]},
        }
    )
    keys = parse_gt_keys(s)
    assert "age" in keys and "gender" in keys


def test_summarize_indexed() -> None:
    o = {
        "0": {"bbox": [1, 2, 3, 4], "age": 25, "emotion": "neutral", "gender": "Woman", "race": "asian"},
    }
    s = json.dumps(o)
    sm = summarize_ground_truth_string(s)
    assert sm.ok
    assert sm.age == 25
    assert sm.emotion == "neutral"
    assert sm.has_bbox
    assert sm.face_count == 1


def test_summarize_multi_age_list() -> None:
    s = json.dumps(
        {
            "0": {"age": 20, "gender": "Woman"},
            "1": {"age": 30, "gender": "Man"},
        }
    )
    sm = summarize_ground_truth_string(s)
    assert sm.ok
    assert sm.ages == [20, 30]
    assert sm.face_count == 2


def test_rejects_non_index_top_level() -> None:
    sm = summarize_ground_truth_string(json.dumps({"not_an_index": {}}))
    assert not sm.ok


def test_summarize_invalid_json() -> None:
    sm = summarize_ground_truth_string("not json")
    assert not sm.ok
    assert sm.error
