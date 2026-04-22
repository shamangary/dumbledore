import pytest

from dumbledore.pipeline_config import DatasetHubConfig, GroundTruthOutputConfig, PipelineConfig, PromptConfig
from dumbledore.prompts import (
    build_training_prompt,
    dataset_prompt_key,
    get_effective_prompt_config,
    image_id_for_path,
)


def test_image_id_for_path_is_stable() -> None:
    a = image_id_for_path("/tmp/a/b.png")
    b = image_id_for_path("/tmp/a/b.png")
    assert a == b
    assert a.startswith("img_")


def test_build_default_matches_gt_schema() -> None:
    g = GroundTruthOutputConfig(age=True, gender=True, emotion=False, race=False, bbox=False, is_real=False)
    p1 = build_training_prompt("img_x", g, None)
    from dumbledore.gt_schema import _full_prompt_text

    p2 = _full_prompt_text("img_x", g, None)
    assert p1 == p2


def test_custom_user_only() -> None:
    g = GroundTruthOutputConfig(age=True, gender=False, emotion=False, race=False, bbox=False, is_real=False)
    pc = PromptConfig(
        system_instruction="SYS",
        user="Hello {image_id} say JSON.",
    )
    out = build_training_prompt("img_abc", g, pc)
    assert out.startswith("SYS\n\n")
    assert "img_abc" in out
    assert "Hello" in out


def test_custom_system_default_user() -> None:
    g = GroundTruthOutputConfig(age=True, gender=False, emotion=False, race=False, bbox=False, is_real=False)
    out = build_training_prompt("img_z", g, PromptConfig(system_instruction="ONLY_SYS", user=None))
    assert out.startswith("ONLY_SYS\n\n")
    assert "For image id" in out  # default user


def test_invalid_placeholder() -> None:
    g = GroundTruthOutputConfig(age=True, gender=False, emotion=False, race=False, bbox=False, is_real=False)
    with pytest.raises(KeyError):
        build_training_prompt("img_x", g, PromptConfig(user="bad {nope}"))


def test_auto_user_includes_dataset_note() -> None:
    g = GroundTruthOutputConfig(age=True, gender=False, emotion=False, race=False, bbox=True, is_real=False)
    w = build_training_prompt("img_x", g, None, dataset_key="wider_face")
    assert "WIDER FACE" in w
    l = build_training_prompt("img_x", g, None, dataset_key="lfw")
    assert "LFW" in l


def test_get_effective_prompt_by_dataset() -> None:
    base = PipelineConfig(
        dataset=DatasetHubConfig(name="wider_face"),
        prompt=PromptConfig(
            system_instruction="BASE_SYS",
            user="BASE_USER",
            by_dataset={
                "wider_face": PromptConfig(system_instruction="W_SYS", user=None),
            },
        ),
    )
    eff = get_effective_prompt_config(base)
    assert eff.system_instruction == "W_SYS"
    assert eff.user == "BASE_USER"


def test_dataset_prompt_key() -> None:
    assert dataset_prompt_key("WIDER") == "wider_face"
    assert dataset_prompt_key("lfw_people") == "lfw"
