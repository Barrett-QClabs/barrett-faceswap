"""Phase 3: swapper catalog, pixel boost, face selection, blend, jobs, pipe."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.blend import apply_color, paste_back, paste_poisson
from barrett_faceswap.cli import main
from barrett_faceswap.compliance.content_filter import FilterDecision
from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.config import ModelFile, apply_request_overrides, load_config
from barrett_faceswap.errors import ConfigError, NoFaceError
from barrett_faceswap.facesel import FaceTracker, assign_slots, bbox_iou, load_face_map
from barrett_faceswap.jobs import JobStore, open_for_run
from barrett_faceswap.media.ffmpeg import build_decode_pipe_command, build_encode_pipe_command, even_size
from barrett_faceswap.model_registry import BUILTIN, ROLE_ORDER
from barrett_faceswap.models import ModelStore, clear_session_cache
from barrett_faceswap.pixel_boost import run_tiled
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.processors.face_swapper import FaceSwapper
from barrett_faceswap.swap import GraphSwapper
from barrett_faceswap.swap_catalog import SWAPPERS, output_size, swapper_by_name
from barrett_faceswap.types import Face, ProcessContext, SwapRequest
from barrett_faceswap.vision import read_image
from tests.conftest import write_config, write_consent, write_png
from tests.onnx_graphs import write_swap_identity, write_vector_identity


def _ffmpeg_or_skip() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is not installed")


def _make_clip(path: Path, *, seconds: float = 0.4, size: str = "160x120") -> None:
    _ffmpeg_or_skip()
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s={size}:r=10:d={seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _config(tmp_path: Path, *, enabled: bool = False):
    path = write_config(tmp_path, enabled=enabled)
    path.write_text(path.read_text(encoding="utf-8") + f"\njobs:\n  root: {tmp_path / 'jobs'}\n", encoding="utf-8")
    return load_config(path)


def _face(x: float, size: float = 40.0, score: float = 0.9) -> Face:
    landmarks = np.array(
        [
            [x + 12, 40],
            [x + 28, 40],
            [x + 20, 55],
            [x + 14, 70],
            [x + 26, 70],
        ],
        dtype=np.float32,
    )
    return Face(
        bbox=np.array([x, 10, x + size, 10 + size], dtype=np.float32),
        score=score,
        landmarks=landmarks,
    )


class _Masks:
    def compose(self, crop, side, blur, landmarks=None, **_ignored):
        del crop, blur, landmarks, _ignored
        return np.ones((side, side), dtype=np.float32)


class _Backend:
    def __init__(self) -> None:
        self.forwards = 0
        self.latent = np.zeros(512, dtype=np.float32)
        self.latent[0] = 1.0

    def embed(self, image, landmarks):
        del image, landmarks
        return self.latent.copy()

    def forward(self, tile, latent):
        del latent
        self.forwards += 1
        return tile


def test_catalog_licenses_and_boost_sizes() -> None:
    names = {
        "inswapper_128",
        "hyperswap_1a_256",
        "hyperswap_1b_256",
        "hyperswap_1c_256",
        "simswap_256",
        "simswap_unofficial_512",
        "ghost_1_256",
        "ghost_2_256",
        "ghost_3_256",
    }
    assert set(SWAPPERS) == names
    assert SWAPPERS["inswapper_128"].crossface_role is None
    assert SWAPPERS["inswapper_128"].embedding == "emap"
    assert SWAPPERS["hyperswap_1a_256"].crossface_role == "crossface_hyperswap"
    assert SWAPPERS["simswap_unofficial_512"].crossface_role == "crossface_simswap"
    assert SWAPPERS["ghost_3_256"].crossface_role == "crossface_ghost"
    assert SWAPPERS["simswap_256"].license_key == "models.license.simswap"
    for spec in SWAPPERS.values():
        assert spec.commercial_use is False
        assert spec.download_url
    for role in ("hyperswap_1a", "simswap_512", "ghost_1", "crossface_hyperswap"):
        assert role in ROLE_ORDER
        assert BUILTIN[role].commercial_use is False
    assert all(record.commercial_use is False for record in BUILTIN.values())
    assert output_size(SWAPPERS["inswapper_128"], pixel_boost=0, resolution=32) == 32
    assert output_size(SWAPPERS["hyperswap_1a_256"], pixel_boost=768, resolution=256) == 768
    with pytest.raises(ConfigError):
        output_size(SWAPPERS["simswap_unofficial_512"], pixel_boost=768, resolution=512)
    with pytest.raises(ConfigError):
        swapper_by_name("not-a-model")


def test_pixel_boost_stitches_native_tiles() -> None:
    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    crop[:8, :8] = 10
    crop[:8, 8:] = 20
    crop[8:, :8] = 30
    crop[8:, 8:] = 40

    def infer(tile: np.ndarray) -> np.ndarray:
        return np.full_like(tile, int(tile[0, 0, 0]) + 1)

    out = run_tiled(crop, 8, infer)
    assert out[0, 0, 0] == 11
    assert out[0, 8, 0] == 21
    assert out[8, 0, 0] == 31
    assert out[8, 8, 0] == 41
    assert run_tiled(crop[:8, :8], 8, infer)[0, 0, 0] == 11


def test_graph_swapper_identity_and_crossface(tmp_path: Path) -> None:
    clear_session_cache()
    swap_path = tmp_path / "hyperswap_1a_256.onnx"
    cross_path = tmp_path / "crossface_hyperswap.onnx"
    write_swap_identity(swap_path, 8)
    write_vector_identity(cross_path)
    config = _config(tmp_path)
    models = config.models.model_copy(
        update={
            "hyperswap_1a": ModelFile(path=swap_path),
            "crossface_hyperswap": ModelFile(path=cross_path),
        }
    )
    config = config.model_copy(update={"models": models})
    spec = swapper_by_name("hyperswap_1a_256")
    backend = GraphSwapper(ModelStore(config, ["cpu"]), spec)
    vector = np.zeros(512, dtype=np.float32)
    vector[3] = 4.0
    projected = backend.project(vector)
    assert projected.shape == (512,)
    assert projected[3] == pytest.approx(1.0, abs=1e-5)
    crop = np.full((8, 8, 3), (40, 90, 160), dtype=np.uint8)
    restored = backend.forward(crop, projected)
    assert restored.shape == crop.shape
    assert int(np.abs(restored.astype(np.int16) - crop.astype(np.int16)).max()) <= 2


def test_tracker_and_assignment() -> None:
    tracker = FaceTracker(iou_threshold=0.3)
    first = tracker.update([np.array([0, 0, 10, 10], dtype=np.float32)])
    first[0].slot = 1
    second = tracker.update([np.array([1, 1, 11, 11], dtype=np.float32)])
    assert second[0].track_id == first[0].track_id
    assert second[0].slot == 1
    third = tracker.update([np.array([1, 1, 11, 11], dtype=np.float32), np.array([40, 40, 50, 50], dtype=np.float32)])
    assert third[0].slot == 1
    assert third[1].track_id != third[0].track_id
    assert bbox_iou(np.array([0, 0, 10, 10]), np.array([40, 40, 50, 50])) == 0.0
    left = np.zeros(4, dtype=np.float32)
    left[0] = 1
    right = np.zeros(4, dtype=np.float32)
    right[1] = 1
    chosen = assign_slots([left, right], [left], 0.45, one_to_one=True)
    assert chosen == [0, None]


def test_color_transfer_and_poisson_shape() -> None:
    swapped = np.zeros((32, 32, 3), dtype=np.uint8)
    swapped[:, :] = (0, 0, 220)
    target = np.zeros_like(swapped)
    target[:, :] = (220, 40, 0)
    mask = np.ones((32, 32), dtype=np.float32)
    lab = apply_color("lab", swapped, target, mask)
    hist = apply_color("histogram", swapped, target, mask)
    assert not np.array_equal(lab, swapped)
    assert not np.array_equal(hist, swapped)
    assert apply_color("none", swapped, target, mask) is swapped
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[:, :] = (20, 20, 20)
    matrix = np.array([[1, 0, 16], [0, 1, 16]], dtype=np.float32)
    cloned = paste_poisson(base, swapped, matrix, mask)
    assert cloned.shape == base.shape
    empty = paste_poisson(base, swapped, matrix, np.zeros_like(mask))
    alpha = paste_back(base, swapped, matrix, np.zeros_like(mask))
    assert np.array_equal(empty, alpha)


def test_many_faces_and_reference_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "swap": config.swap.model_copy(update={"selector": "many", "resolution": 32}),
            "blend": config.blend.model_copy(update={"color": "none"}),
        }
    )
    swapper = FaceSwapper(config, ["cpu"])
    backend = _Backend()
    swapper.store = ModelStore(config, ["cpu"])
    swapper.masks = _Masks()
    swapper._backend = lambda: backend  # type: ignore[method-assign]
    faces = [_face(10, size=80), _face(140, size=50)]
    swapper._detect = lambda image: faces  # type: ignore[method-assign]
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    ctx = ProcessContext(source_bgr=frame, dry_run=False)
    swapper.process_frame(frame, ctx)
    assert backend.forwards == 2
    assert "faces=2" in ctx.notes[-1] or "2" in ctx.notes[-1]

    def fake_embed(store, image, landmarks):
        del store, image
        vector = np.zeros(512, dtype=np.float32)
        vector[0 if float(np.asarray(landmarks)[0, 0]) < 80 else 1] = 1.0
        return vector

    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.embed_arcface", fake_embed)
    reference = config.model_copy(
        update={"swap": config.swap.model_copy(update={"selector": "reference", "similarity": 0.45, "track": True})}
    )
    picker = FaceSwapper(reference, ["cpu"])
    other = _Backend()
    picker.store = swapper.store
    picker.masks = _Masks()
    picker._backend = lambda: other  # type: ignore[method-assign]
    picker._detect = lambda image: faces  # type: ignore[method-assign]
    picker.process_frame(frame, ProcessContext(source_bgr=frame, dry_run=False, log_every_frame=False))
    assert other.forwards == 1
    picker.process_frame(frame, ProcessContext(source_bgr=frame, dry_run=False, log_every_frame=False))
    assert other.forwards == 2
    assert picker.tracker.tracks[0].slot == 0


def test_face_map_locks_an_index(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source, size=(80, 80))
    payload = [{"source": str(source), "target_index": 1}]
    face_map = tmp_path / "map.json"
    face_map.write_text(json.dumps(payload), encoding="utf-8")
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "swap": config.swap.model_copy(update={"selector": "map", "face_map": face_map, "resolution": 32}),
            "blend": config.blend.model_copy(update={"color": "none"}),
        }
    )
    swapper = FaceSwapper(config, ["cpu"])
    backend = _Backend()
    swapper.store = ModelStore(config, ["cpu"])
    swapper.masks = _Masks()
    swapper._backend = lambda: backend  # type: ignore[method-assign]
    faces = [_face(10, size=90), _face(150, size=40)]
    swapper._detect = lambda image: list(faces)  # type: ignore[method-assign]
    frame = np.zeros((200, 260, 3), dtype=np.uint8)
    ctx = ProcessContext(source_bgr=read_image(source), dry_run=False, log_every_frame=False)
    ordered = sorted(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])), reverse=True)
    pairs = swapper._pairs(frame, ordered, swapper._identities(ctx), ctx)
    assert len(pairs) == 1
    assert pairs[0][0] is ordered[1]
    with pytest.raises(ConfigError):
        load_face_map(str(tmp_path / "missing.json"))


def test_overrides_keep_blend_and_select_model(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = SwapRequest(
        source_path=tmp_path / "a.png",
        target_path=tmp_path / "b.png",
        output_path=tmp_path / "c.png",
        swapper_model="ghost_2_256",
        pixel_boost=512,
        blend_mode="poisson",
        color_transfer="lab",
        many_faces=True,
        video_mode="pipe",
        enhancer_blend=0.4,
    )
    updated = apply_request_overrides(config, request)
    assert updated.swap.model == "ghost_2_256"
    assert updated.swap.pixel_boost == 512
    assert updated.swap.selector == "many"
    assert updated.blend.mode == "poisson"
    assert updated.blend.color == "lab"
    assert updated.enhancer.blend == 0.4
    assert updated.video.mode == "pipe"
    assert updated.compliance.enabled is False


def test_image_job_and_resume_guard(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    output = tmp_path / "out.png"
    write_png(source)
    write_png(target, color=(10, 20, 30))
    config = _config(tmp_path)
    request = SwapRequest(
        source_path=source,
        target_path=target,
        output_path=output,
        dry_run=True,
        job_id="still1",
    )
    result = FaceSwapPipeline(config).run(request)
    assert output.is_file()
    state = JobStore(config.jobs.root).load("still1")
    assert state.state == "done"
    assert state.next_frame == 1
    assert result.compliance_enabled is False
    with pytest.raises(ConfigError):
        open_for_run(JobStore(config.jobs.root), request)
    code = main(
        [
            "jobs",
            "list",
            "--config",
            str(tmp_path / "config.yaml"),
        ]
    )
    assert code == 0


def test_disk_job_resumes_after_the_last_finished_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "out.mp4"
    write_png(source, size=(160, 120))
    _make_clip(target, seconds=0.4)
    config = _config(tmp_path)
    writes = {"n": 0}
    real_write = __import__("barrett_faceswap.video", fromlist=["write_image"]).write_image

    def stop_early(path, image):
        writes["n"] += 1
        if writes["n"] >= 2:
            raise RuntimeError("stop")
        return real_write(path, image)

    monkeypatch.setattr("barrett_faceswap.video.write_image", stop_early)
    request = SwapRequest(
        source_path=source,
        target_path=target,
        output_path=output,
        dry_run=True,
        job_id="clip1",
        video_threads=2,
    )
    with pytest.raises(RuntimeError):
        FaceSwapPipeline(config).run(request)
    store = JobStore(config.jobs.root)
    state = store.load("clip1")
    assert state.state == "interrupted"
    assert state.next_frame == 1
    first = sorted((store.directory("clip1") / "processed").glob("frame_*.png"))[0]
    before = first.read_bytes()
    monkeypatch.setattr("barrett_faceswap.video.write_image", real_write)
    resumed = request.model_copy(update={"resume": True})
    FaceSwapPipeline(config).run(resumed)
    assert first.read_bytes() == before
    assert output.is_file()
    assert store.load("clip1").state == "done"
    with pytest.raises(ConfigError):
        FaceSwapPipeline(config).run(request.model_copy(update={"resume": False}))


def test_pipe_dry_run_and_compliance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ffmpeg_or_skip()
    source = tmp_path / "source.png"
    target = tmp_path / "clip.mp4"
    output = tmp_path / "piped.mp4"
    write_png(source, size=(160, 120))
    _make_clip(target, seconds=0.3)
    config = _config(tmp_path, enabled=False)
    result = FaceSwapPipeline(config).run(
        SwapRequest(
            source_path=source,
            target_path=target,
            output_path=output,
            dry_run=True,
            video_mode="pipe",
        )
    )
    assert output.is_file()
    assert result.compliance_enabled is False
    assert any("pipe" in note.lower() or "管道" in note for note in result.notes)

    write_consent(tmp_path / "consent.json")
    (tmp_path / "audit-case").mkdir()
    audited = _config(tmp_path / "audit-case", enabled=True)
    calls: list[str] = []

    class _Counter:
        name = "counter"
        supports_minor_detection = False

        def assess(self, image, *, role: str) -> FilterDecision:
            del image
            calls.append(role)
            return FilterDecision(allowed=True, provider="stub", reason="ok", minor_detection="unavailable")

    monkeypatch.setattr(
        "barrett_faceswap.compliance.policy.content_filter.create_filter",
        lambda _settings: _Counter(),
    )
    befores = {"count": 0}
    original = CompliancePolicy.before_process

    def _counting(self, request, **kwargs):
        befores["count"] += 1
        return original(self, request, **kwargs)

    monkeypatch.setattr(CompliancePolicy, "before_process", _counting)
    marked = tmp_path / "marked.mp4"
    FaceSwapPipeline(audited, admin_path=tmp_path / "missing.toml").run(
        SwapRequest(
            source_path=source,
            target_path=target,
            output_path=marked,
            dry_run=True,
            enable_compliance=True,
            consent_file=tmp_path / "consent.json",
            video_mode="pipe",
            filter_every_n_frames=1,
        )
    )
    assert befores["count"] == 1
    records = [
        json.loads(line)
        for line in audited.compliance.audit.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["event"] == "completed"
    assert calls[0] == "source"
    assert any(role.startswith("output[") for role in calls)
    still = tmp_path / "still.png"
    write_png(still)
    with pytest.raises(ConfigError):
        FaceSwapPipeline(config).run(
            SwapRequest(
                source_path=source,
                target_path=still,
                output_path=tmp_path / "nope.png",
                dry_run=True,
                video_mode="pipe",
            )
        )


def test_pipe_commands_keep_the_named_encoder(tmp_path: Path) -> None:
    decode = build_decode_pipe_command(
        tmp_path / "in.mp4",
        width=160,
        height=120,
        trim_start=0.5,
        trim_end=1.5,
        seek=0.2,
    )
    assert decode[0].endswith("ffmpeg") or "ffmpeg" in decode[0]
    assert "pipe:1" in decode
    assert "bgr24" in decode
    assert "-ss" in decode
    encode = build_encode_pipe_command(
        tmp_path / "out.mp4",
        width=161,
        height=119,
        fps=10,
        encoder="h264_nvenc",
        crf=23,
        audio_source=None,
        audio_mode="none",
        trim_start=None,
        trim_end=None,
    )
    assert "h264_nvenc" in encode
    assert "libx264" not in encode
    assert "pipe:0" in encode
    assert even_size(161, 119) == (160, 118)


def test_no_match_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={"swap": config.swap.model_copy(update={"selector": "reference", "similarity": 0.99, "track": False})}
    )

    def far(store, image, landmarks):
        del store, image, landmarks
        vector = np.zeros(512, dtype=np.float32)
        vector[2] = 1.0
        return vector

    monkeypatch.setattr("barrett_faceswap.processors.face_swapper.embed_arcface", far)
    swapper = FaceSwapper(config, ["cpu"])
    backend = _Backend()
    swapper.store = ModelStore(config, ["cpu"])
    swapper.masks = _Masks()
    swapper._backend = lambda: backend  # type: ignore[method-assign]
    swapper._detect = lambda image: [_face(20)]  # type: ignore[method-assign]
    with pytest.raises(NoFaceError):
        swapper.process_frame(
            np.zeros((120, 160, 3), dtype=np.uint8),
            ProcessContext(source_bgr=np.zeros((40, 40, 3), dtype=np.uint8), dry_run=False),
        )
