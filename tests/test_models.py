"""Hash pins and the inference-session cache key."""

from __future__ import annotations

from pathlib import Path

import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ModelHashError
from barrett_faceswap.model_registry import BUILTIN, describe_models
from barrett_faceswap.models import ModelStore, clear_session_cache, file_sha256, verify_model_hash
from tests.conftest import REPO, write_config


def test_registry_marks_insightface_weights_non_commercial() -> None:
    for role in ("detector", "recognizer", "swapper"):
        record = BUILTIN[role]
        assert record.commercial_use is False
        assert record.sha256 is None
        assert record.input_size[0] > 0
    assert BUILTIN["swapper"].alignment == "inswapper_128"
    assert BUILTIN["recognizer"].alignment == "arcface_112"
    assert BUILTIN["detector"].alignment == "scrfd"


def test_hash_pin_accepts_a_match_and_rejects_a_mismatch(tmp_path: Path) -> None:
    weight = tmp_path / "det_10g.onnx"
    weight.write_bytes(b"not-a-real-model")
    digest = file_sha256(weight)
    assert verify_model_hash(weight, digest, "detector") == digest
    with pytest.raises(ModelHashError):
        verify_model_hash(weight, "ab" * 32, "detector")


def test_session_cache_reuses_the_same_model_provider_and_device(tmp_path: Path) -> None:
    weight = tmp_path / "weights"
    weight.mkdir()
    detector = weight / "det_10g.onnx"
    detector.write_bytes(b"detector-bytes")
    config_path = write_config(tmp_path, enabled=False)
    # Point the config at the file we just wrote. write_config already uses tmp/weights.
    config = load_config(config_path)
    clear_session_cache()
    calls: list[int] = []

    def factory(path: Path, providers, device_id: int):
        del path, providers, device_id
        calls.append(1)
        return {"session": len(calls)}

    store = ModelStore(config, providers=["cpu"])
    first = store.session("detector", factory=factory)
    second = store.session("detector", factory=factory)
    assert first is second
    assert calls == [1]
    other = ModelStore(config, providers=["cpu"])
    other.device_id = 1
    other.session("detector", factory=factory)
    assert calls == [1, 1]


def test_inspect_lines_mention_the_license(tmp_path: Path) -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    lines = describe_models(config)
    assert len(lines) == 3
    assert any("inswapper_128" in line for line in lines)
    assert any("non-commercial" in line or "非商业" in line for line in lines)
