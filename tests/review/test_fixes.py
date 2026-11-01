"""Synthetic regression coverage for the confirmed review findings."""
import atexit
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.compliance.watermark import save_image
from barrett_faceswap.config import load_config
from barrett_faceswap.errors import ContentBlockedError, NoFaceError
from barrett_faceswap.facesel import FaceMapEntry, Track
from barrett_faceswap.library import FaceLibrary
from barrett_faceswap.output import reserve_output
from barrett_faceswap.processors.face_swapper import FaceSwapper, _Slot
from barrett_faceswap.types import Face, ProcessContext, SwapRequest
from tests.conftest import write_config, write_png


@pytest.fixture
def config(tmp_path):
    return load_config(write_config(tmp_path, enabled=False))


def face(offset=0):
    return Face(bbox=np.array([offset, 0, offset + 10, 10], dtype=np.float32), score=.9,
                landmarks=np.ones((5, 2), dtype=np.float32))


def test_library_rejects_symlink_without_reading_external_profile(tmp_path):
    lib = FaceLibrary(tmp_path / "library")
    lib.create("safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "profile.json").write_text('{"id":"alias","name":"alias"}')
    (lib.root / "alias").symlink_to(outside, target_is_directory=True)
    assert lib._find_dir("alias") is None
    with pytest.raises(Exception):
        lib.delete("alias")
    assert outside.is_dir()


@pytest.mark.parametrize("key", ["../outside", "/tmp/outside", "a/b", "a\\b", ".", "..", "bad:key"])
def test_library_invalid_keys_do_not_read_profiles(tmp_path, key):
    lib = FaceLibrary(tmp_path / "library")
    lib.create("safe")
    assert lib._find_dir(key) is None


def test_image_returns_unique_path_and_keeps_existing_sidecar(tmp_path):
    original = tmp_path / "out.png"
    original.write_bytes(b"original")
    sidecar = tmp_path / "out-1.png.meta.json"
    sidecar.write_text("keep")
    result = save_image(np.zeros((8, 8, 3), np.uint8), original, watermark=True, text="AI")
    assert result.name == "out-2.png"
    assert original.read_bytes() == b"original"
    assert sidecar.read_text() == "keep"
    assert result.is_file()
    assert not list(tmp_path.glob(".barrett-*"))


def test_exclusive_output_allocations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(reserve_output, [tmp_path / "out.mp4"] * 16))
    assert len(set(paths)) == 16


@pytest.mark.parametrize("selector", ["index", "map"])
def test_missing_selected_face_is_optional(config, selector):
    config.swap.selector = selector
    config.swap.track = False
    processor = FaceSwapper(config, ["cpu"])
    ctx = ProcessContext(np.zeros((8, 8, 3), np.uint8), False, target_face_index=1, allow_missing_face=True)
    slots = [_Slot(np.ones(4), np.ones(4), target_index=1)]
    assert processor._pairs(ctx.source_bgr, [face()], slots, ctx) == []
    ctx.allow_missing_face = False
    with pytest.raises(NoFaceError):
        processor._pairs(ctx.source_bgr, [face()], slots, ctx)


def test_tracked_slot_cannot_be_assigned_twice(config, monkeypatch):
    config.swap.selector = "map"
    config.swap.track = True
    processor = FaceSwapper(config, ["cpu"])
    faces = [face(), face(20)]
    tracks = [Track(1, faces[0].bbox, slot=0), Track(2, faces[1].bbox)]
    monkeypatch.setattr(processor.tracker, "update", lambda boxes: tracks)
    monkeypatch.setattr(processor, "_arc_of", lambda *args: np.array([1., 0.]))
    slots = [_Slot(np.ones(2), np.array([1., 0.])), _Slot(np.ones(2), np.array([1., 0.]))]
    ctx = ProcessContext(np.zeros((8, 8, 3), np.uint8), False)
    pairs = processor._pairs(ctx.source_bgr, faces, slots, ctx)
    assert len(pairs) == 2
    assert pairs[0][1] is slots[0] and pairs[1][1] is slots[1]


@pytest.mark.parametrize("reference", [None, "reference.png"])
def test_blendswap_matching_uses_arcface_vectors(config, monkeypatch, reference):
    config.swap.model = "blendswap_256"
    processor = FaceSwapper(config, ["cpu"])
    image = np.zeros((8, 8, 3), np.uint8)
    crop = np.zeros((1, 3, 112, 112), np.float32)
    vector = np.array([1., 0.], np.float32)
    monkeypatch.setattr(processor, "_backend", Mock(return_value=object()))
    monkeypatch.setattr(processor, "_map_entries", lambda ctx: [FaceMapEntry("", reference=reference)])
    monkeypatch.setattr(processor, "_embed_image", lambda *args: (crop, .9))
    monkeypatch.setattr(processor, "_detect", lambda image: [face()])
    monkeypatch.setattr(processor, "_arc_of", lambda *args: vector)
    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.read_image", lambda path: image)
    ctx = ProcessContext(image, False, source_embedding=np.ones(2))
    slot = processor._identities(ctx)[0]
    assert slot.latent is crop
    assert slot.match.shape == (2,)
    assert np.array_equal(slot.match, vector)


def test_live_missing_face_passes_through(config, tmp_path, monkeypatch):
    from barrett_faceswap.live import LiveSession
    source = tmp_path / "source.png"
    write_png(source)
    request = SwapRequest(source_path=source, target_path=Path("camera"), output_path=tmp_path / "session.txt")
    processor = FaceSwapper(config, ["cpu"])
    processor.store = Mock()
    monkeypatch.setattr(processor, "pre_check", lambda ctx: None)
    monkeypatch.setattr(processor, "prepare_source", lambda ctx: None)
    monkeypatch.setattr(processor, "_detector", lambda: Mock(try_detect=Mock(return_value=[])))
    session = LiveSession(config, request, processors=[processor])
    image = np.zeros((8, 8, 3), np.uint8)
    assert session.step(image) is image
    assert session.step(image) is image
    assert session._shared.allow_missing_face


@pytest.mark.parametrize("modern", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_swapper_detector_compatibility(config, monkeypatch, modern, missing):
    from types import SimpleNamespace
    processor = FaceSwapper(config, ["cpu"])
    image = np.zeros((8, 8, 3), np.uint8)
    detected = [face()]
    detect = Mock(side_effect=NoFaceError("no face")) if missing else Mock(return_value=detected)
    detector = SimpleNamespace(**{"try_detect" if modern else "detect": detect})
    if modern:
        detector.detect = Mock(side_effect=AssertionError("try_detect should be preferred"))
    monkeypatch.setattr(processor, "_detector", lambda: detector)
    assert processor._detect(image) == ([] if missing else detected)
    detect.assert_called_once_with(image)
    processor.store = Mock()
    if missing:
        ctx = ProcessContext(image, False, allow_missing_face=True)
        assert processor.process_frame(image, ctx) is image
        ctx.allow_missing_face = False
        with pytest.raises(NoFaceError):
            processor.process_frame(image, ctx)


def test_swapper_detector_preserves_unexpected_errors(config, monkeypatch):
    from types import SimpleNamespace
    processor = FaceSwapper(config, ["cpu"])
    monkeypatch.setattr(processor, "_detector", lambda: SimpleNamespace(
        detect=Mock(side_effect=RuntimeError("detector failure"))))
    with pytest.raises(RuntimeError, match="detector failure"):
        processor._detect(np.zeros((8, 8, 3), np.uint8))


@pytest.mark.parametrize("blocked", [False, True])
def test_compliance_screens_and_audits_mapping_library_source(tmp_path, monkeypatch, blocked):
    config = load_config(write_config(tmp_path, enabled=True))
    source, target, extra = [tmp_path / name for name in ("source.png", "target.png", "library-still.png")]
    for path in (source, target, extra):
        write_png(path)
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps([{"source": str(source), "target_index": 0},
                                   {"source": str(extra), "embedding": [1., 0.], "target_index": 1}]))
    config.swap.selector = "map"
    config.swap.face_map = mapping
    policy = CompliancePolicy(config)
    from barrett_faceswap.compliance.content_filter import FilterDecision
    checker = Mock()
    checker.assess.side_effect = lambda image, role: FilterDecision(
        allowed=not (blocked and role == "source[1]"), provider="mock", reason="synthetic verdict")
    monkeypatch.setattr("barrett_faceswap.compliance.policy.content_filter.create_filter", lambda settings: checker)
    request = SwapRequest(source_path=source, target_path=target, output_path=tmp_path / "out.png",
                          acknowledge_authorized_use=True, consent_subject="synthetic",
                          consent_statement="Synthetic test patterns only; authorized for testing.")
    if blocked:
        with pytest.raises(ContentBlockedError):
            policy.before_process(request)
    else:
        context = policy.before_process(request)
        policy.after_process(context, np.zeros((8, 8, 3), np.uint8), request)
    event = json.loads(config.compliance.audit.path.read_text().splitlines()[-1])
    assert event["event"] == ("refused" if blocked else "completed")
    assert [item["path"] for item in event["sources"]] == [str(source), str(extra)]
    assert all(item["sha256"] for item in event["sources"])
    assert "source[1]" in [item["role"] for item in event["content_filter"]["decisions"]]


def test_blendswap_library_video_import(config, tmp_path, monkeypatch):
    from barrett_faceswap.library import _parts_from_video
    config.swap.model = "blendswap_256"
    image = np.zeros((8, 8, 3), np.uint8)
    # Exercise the fallback for a prepared crop without a persisted vector.
    monkeypatch.setattr("barrett_faceswap.library.prepare_source", lambda cfg, req, work: (req, []))
    monkeypatch.setattr("barrett_faceswap.library.read_image", lambda path: image)
    monkeypatch.setattr("barrett_faceswap.library.embed_still", lambda cfg, img: (np.array([1., 0.]), .9, img))
    monkeypatch.setattr("barrett_faceswap.library.write_speech_reference", lambda *args: False)
    parts = _parts_from_video(config, tmp_path / "synthetic.mp4")
    assert np.array_equal(parts.embedding, [1., 0.])
    assert parts.still is image


def test_mapper_replacement_and_exit_cleanup():
    from barrett_faceswap.ui.map_workspace import MapWorkspace
    workspace = MapWorkspace()
    first = workspace.reset()
    (first / "crop.png").write_bytes(b"synthetic")
    second = workspace.reset()
    assert not first.exists() and second.is_dir()
    workspace.cleanup()
    workspace.cleanup()
    assert not second.exists()
    atexit.unregister(workspace.cleanup)


@pytest.mark.parametrize("mode", ["disk", "pipe"])
def test_video_failure_only_removes_own_reservation(config, tmp_path, monkeypatch, mode):
    from barrett_faceswap.video import VideoPipeline
    source, target, output = [tmp_path / name for name in ("source.png", "target.mp4", "existing.mp4")]
    source.write_bytes(b"synthetic")
    target.write_bytes(b"synthetic")
    output.write_bytes(b"KEEP")
    config.video.mode = mode
    def fail_probe(path):
        raise RuntimeError("synthetic probe failure")
    monkeypatch.setattr("barrett_faceswap.video.probe", fail_probe)
    with pytest.raises(RuntimeError, match="probe failure"):
        VideoPipeline(config).run(SwapRequest(source_path=source, target_path=target, output_path=output))
    assert output.read_bytes() == b"KEEP"
    assert not (tmp_path / "existing-1.mp4").exists()
    assert not list(tmp_path.glob(".barrett-*"))


@pytest.mark.parametrize("mode", ["disk", "pipe"])
def test_video_collision_returns_actual_output(config, tmp_path, mode):
    from tests.test_video import _make_clip
    from barrett_faceswap.pipeline import FaceSwapPipeline
    source, target, output = [tmp_path / name for name in ("source.png", "target.mp4", "existing.mp4")]
    write_png(source)
    _make_clip(target, size="32x32")
    output.write_bytes(b"KEEP")
    config.video.mode = mode
    result = FaceSwapPipeline(config).run(SwapRequest(source_path=source, target_path=target,
                                                      output_path=output, dry_run=True))
    assert output.read_bytes() == b"KEEP"
    assert result.output_path == tmp_path / "existing-1.mp4"
    assert result.output_path.stat().st_size > 0
    assert not list(tmp_path.glob(".barrett-*"))


def test_encoder_failure_never_removes_competing_output(tmp_path, monkeypatch):
    from barrett_faceswap.media import ffmpeg
    output = tmp_path / "out.mp4"
    def fail_encode(frames, work, **options):
        output.write_bytes(b"another writer")
        work.write_bytes(b"partial")
        raise RuntimeError("synthetic encoder failure")
    monkeypatch.setattr(ffmpeg, "_encode_video", fail_encode)
    with pytest.raises(RuntimeError):
        ffmpeg.encode_video(tmp_path, output)
    assert output.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".barrett-*"))


def test_encoder_publication_is_exclusive(tmp_path, monkeypatch):
    from barrett_faceswap.media import ffmpeg
    output = tmp_path / "out.mp4"
    def encode(frames, work, **options):
        output.write_bytes(b"another writer")
        work.write_bytes(b"new encoded frames")
        return "none"
    monkeypatch.setattr(ffmpeg, "_encode_video", encode)
    with pytest.raises(FileExistsError):
        ffmpeg.encode_video(tmp_path, output)
    assert output.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".barrett-*"))


def test_library_display_name_with_spaces_still_resolves(tmp_path):
    lib = FaceLibrary(tmp_path / "library")
    profile = lib.create("Ada Lovelace")
    assert lib._find_dir("Ada Lovelace") == lib.root / profile.id
    with pytest.raises(Exception):
        lib.create("Ada Lovelace")
    lib.delete("Ada Lovelace")
    assert not (lib.root / profile.id).exists()
