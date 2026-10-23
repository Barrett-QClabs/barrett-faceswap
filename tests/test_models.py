"""Hash pins and the inference-session cache key."""

from __future__ import annotations

from pathlib import Path

import pytest

from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ModelHashError
from barrett_faceswap.model_registry import BUILTIN, describe_models
import barrett_faceswap.models as model_api
from barrett_faceswap.models import ModelStore, clear_session_cache, file_sha256, verify_model_hash
from tests.conftest import REPO, write_config


def test_registry_marks_insightface_weights_non_commercial() -> None:
    for role, record in BUILTIN.items():
        assert record.commercial_use is False, role
        assert record.sha256 is None
        assert record.input_size[0] > 0
        assert record.license_key
    assert BUILTIN["swapper"].alignment == "inswapper_128"
    assert BUILTIN["recognizer"].alignment == "arcface_112"
    assert BUILTIN["detector"].alignment == "scrfd"
    assert BUILTIN["gfpgan"].alignment == "ffhq_512"
    assert BUILTIN["gfpgan"].filename == "gfpgan_1.4.onnx"
    assert BUILTIN["codeformer"].license_key == "models.license.codeformer"


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


def test_session_hashes_once_per_load_and_again_after_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight = tmp_path / "weights"
    weight.mkdir()
    detector = weight / "det_10g.onnx"
    detector.write_bytes(b"detector-bytes")
    config = load_config(write_config(tmp_path, enabled=False))
    clear_session_cache()
    reads: list[int] = []
    verifies: list[str] = []
    built: list[int] = []

    real_read = model_api._file_sha256_uncached
    real_verify = verify_model_hash

    def counted_read(path: Path) -> str:
        reads.append(1)
        return real_read(path)

    def counted_verify(path: Path, expected: str | None, role: str) -> str:
        verifies.append(role)
        return real_verify(path, expected, role)

    monkeypatch.setattr("barrett_faceswap.models._file_sha256_uncached", counted_read)
    monkeypatch.setattr("barrett_faceswap.models.verify_model_hash", counted_verify)

    assert file_sha256(detector) == file_sha256(detector)
    assert reads == [1]

    def factory(path: Path, providers, device_id: int):
        del path, providers, device_id
        built.append(1)
        return {"session": len(built)}

    store = ModelStore(config, providers=["cpu"])
    first = store.session("detector", factory=factory)
    second = store.session("detector", factory=factory)
    assert first is second
    assert built == [1]
    assert verifies == ["detector"]
    assert reads == [1]

    detector.write_bytes(b"detector-bytes-replaced-and-longer")
    third = store.session("detector", factory=factory)
    assert third is not first
    assert built == [1, 1]
    assert verifies == ["detector", "detector"]
    assert reads == [1, 1]

    pinned = config.models.detector.model_copy(update={"sha256": "ab" * 32})
    refused = config.model_copy(
        update={"models": config.models.model_copy(update={"detector": pinned})}
    )
    clear_session_cache()
    with pytest.raises(ModelHashError):
        ModelStore(refused, providers=["cpu"]).session("detector", factory=factory)
    assert built == [1, 1]


def test_inspect_lines_mention_the_license(tmp_path: Path) -> None:
    config = load_config(REPO / "configs" / "default.yaml")
    lines = describe_models(config)
    from barrett_faceswap.model_registry import ROLE_ORDER

    assert len(lines) == len(ROLE_ORDER)
    assert any("inswapper_128" in line for line in lines)
    assert any("gfpgan_1.4" in line for line in lines)
    assert any("Apache" in line or "S-Lab" in line for line in lines)
    assert any("non-commercial" in line or "非商业" in line for line in lines)
