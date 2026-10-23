"""Phase 4: live loop, execution options, commercial gate, watermark, scaffold.

These tests do not open a webcam, do not need GPU, and do not download weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from barrett_faceswap.binding import run_feeds
from barrett_faceswap.commercial import guard_commercial
from barrett_faceswap.compliance.policy import CompliancePolicy
from barrett_faceswap.compliance.watermark import apply_visible_watermark
from barrett_faceswap.config import TensorRTConfig, load_config
from barrett_faceswap.errors import ConfigError, NoFaceError
from barrett_faceswap.execution import resolve_providers, tensorrt_options
from barrett_faceswap.face_cache import DetectionCache
from barrett_faceswap.i18n import t
from barrett_faceswap.live import LivePipeline, LoopbackWriter
from barrett_faceswap.models import weight_path
from barrett_faceswap.pipeline import FaceSwapPipeline
from barrett_faceswap.queues import LatestQueue
from barrett_faceswap.train import export_contract, identity_cosine, ssim, training_plan, validate_dataset
from barrett_faceswap.types import Face, SwapRequest
from barrett_faceswap.video import watermark_frames
from tests.conftest import REPO, write_config, write_consent, write_png


def test_latest_queue_drops_the_oldest() -> None:
    queue = LatestQueue(1)
    queue.put_latest("a")
    queue.put_latest("b")
    queue.put_latest("c")
    assert queue.dropped == 2
    assert queue.get(0.1) == "c"
    assert queue.get(0.01) is None


def test_detection_cache_interpolates_between_detections() -> None:
    cache = DetectionCache(3)
    landmarks = np.array([[12, 22], [30, 22], [20, 35], [14, 48], [28, 48]], dtype=np.float32)
    first = Face(bbox=np.array([10, 20, 40, 60], np.float32), score=0.9, landmarks=landmarks)
    assert cache.wants_detection()
    cache.observe([first])
    moved = Face(bbox=np.array([14, 20, 44, 60], np.float32), score=0.91, landmarks=landmarks.copy())
    cache.observe([moved])
    assert cache.wants_detection() is False
    advanced = cache.observe(None)
    assert advanced[0].bbox[0] == pytest.approx(18)
    assert advanced[0].bbox[2] == pytest.approx(48)
    assert cache.observe([]) == []
    assert cache.observe(None) == []


class _FakeSwapper:
    def __init__(self) -> None:
        self.prepared = 0
        self.detected = 0
        self.frames = 0

    def pre_check(self, ctx) -> None:
        return None

    def prepare_source(self, ctx) -> None:
        self.prepared += 1

    def detect_faces(self, frame):
        self.detected += 1
        return [
            Face(
                bbox=np.array([8, 8, 40, 40], np.float32),
                score=0.8,
                landmarks=np.zeros((5, 2), np.float32) + 12,
            )
        ]

    def process_frame(self, frame, ctx):
        self.frames += 1
        assert ctx.known_faces is not None
        return frame

    def post_process(self) -> None:
        return None


def _live_config(tmp_path: Path, *, enabled: bool, queue_size: int = 8, detect_every: int = 3):
    config = load_config(write_config(tmp_path, enabled=enabled))
    return config.model_copy(
        update={
            "live": config.live.model_copy(
                update={"queue_size": queue_size, "detect_every": detect_every, "preview": False}
            )
        }
    )


def _request(tmp_path: Path, **overrides) -> SwapRequest:
    source = tmp_path / "source.png"
    write_png(source, (30, 60, 90), (64, 48))
    fields = dict(
        source_path=source,
        target_path=Path("camera-test"),
        output_path=tmp_path / "live-session.txt",
        dry_run=True,
        live_session=True,
        live_frames=4,
    )
    fields.update(overrides)
    return SwapRequest(**fields)


def test_live_pipeline_uses_synthetic_frames(tmp_path: Path) -> None:
    config = _live_config(tmp_path, enabled=False)
    frames = [np.full((48, 64, 3), value, np.uint8) for value in (10, 20, 30, 40)]
    seen: list[np.ndarray] = []
    result, stats = LivePipeline(config, admin_path=tmp_path / "missing.toml").run(
        _request(tmp_path),
        frames,
        writer=seen.append,
        preview=False,
    )
    assert result.compliance_enabled is False
    assert stats.frames_out == 4
    assert len(seen) == 4
    assert not (tmp_path / "audit.jsonl").exists()


def test_source_is_prepared_once_and_detection_is_strided(tmp_path: Path) -> None:
    config = _live_config(tmp_path, enabled=False, detect_every=3)
    fake = _FakeSwapper()
    frames = [np.zeros((32, 32, 3), np.uint8) for _ in range(6)]
    _result, stats = LivePipeline(config, admin_path=tmp_path / "missing.toml").run(
        _request(tmp_path, dry_run=False, live_frames=6),
        frames,
        preview=False,
        processors=[fake],
    )
    assert fake.prepared == 1
    assert fake.detected == 2
    assert fake.frames == 6
    assert stats.frames_out == 6
    assert stats.prepared == 1


def test_live_miss_leaves_the_frame_and_clears_the_cache(tmp_path: Path) -> None:
    config = _live_config(tmp_path, enabled=False, detect_every=1)

    class _Miss(_FakeSwapper):
        def detect_faces(self, frame):
            del frame
            self.detected += 1
            raise NoFaceError("none")

        def process_frame(self, frame, ctx):
            self.frames += 1
            assert list(ctx.known_faces) == []
            return frame

    fake = _Miss()
    frames = [np.full((16, 16, 3), 30, np.uint8) for _ in range(3)]
    seen: list[np.ndarray] = []
    _result, stats = LivePipeline(config, admin_path=tmp_path / "missing.toml").run(
        _request(tmp_path, dry_run=False, live_frames=3),
        frames,
        writer=seen.append,
        preview=False,
        processors=[fake],
    )
    assert stats.frames_out == 3
    assert fake.detected == 3
    assert len(seen) == 3
    assert all(np.array_equal(out, src) for out, src in zip(seen, frames, strict=True))


def test_slow_processor_drops_stale_frames(tmp_path: Path) -> None:
    import time

    class Slow(_FakeSwapper):
        def process_frame(self, frame, ctx):
            time.sleep(0.02)
            return super().process_frame(frame, ctx)

    config = _live_config(tmp_path, enabled=False, queue_size=1, detect_every=1)
    fake = Slow()
    frames = [np.zeros((24, 24, 3), np.uint8) for _ in range(12)]
    _result, stats = LivePipeline(config, admin_path=tmp_path / "missing.toml").run(
        _request(tmp_path, dry_run=False, live_frames=0),
        frames,
        preview=False,
        processors=[fake],
    )
    assert stats.dropped >= 1
    assert stats.frames_out < 12
    assert fake.prepared == 1


def test_live_compliance_watermarks_and_audits_once(tmp_path: Path) -> None:
    config = _live_config(tmp_path, enabled=True)
    consent = tmp_path / "consent.json"
    write_consent(consent)
    frames = [np.full((48, 64, 3), 40, np.uint8) for _ in range(3)]
    seen: list[np.ndarray] = []
    result, stats = LivePipeline(config, admin_path=tmp_path / "missing.toml").run(
        _request(tmp_path, enable_compliance=True, consent_file=consent, live_frames=3),
        frames,
        writer=seen.append,
        preview=False,
    )
    assert result.compliance_enabled is True
    assert stats.frames_out == 3
    assert seen[0].shape == frames[0].shape
    assert not np.array_equal(seen[0], frames[0])
    assert t("live.session_marked") in result.notes
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(audit) == 1
    assert "completed" in audit[0]


def test_missing_virtual_camera_does_not_spawn_ffmpeg(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    writer = LoopbackWriter(str(tmp_path / "missing-video"))
    writer.write(np.zeros((8, 8, 3), np.uint8))
    writer.close()
    assert writer._proc is False
    assert "missing-video" in caplog.text


def test_commercial_flag_refuses_builtin_models(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    write_png(source)
    write_png(target)
    request = SwapRequest(
        source_path=source,
        target_path=target,
        output_path=tmp_path / "out.png",
        dry_run=True,
        commercial=True,
    )
    with pytest.raises(ConfigError) as caught:
        FaceSwapPipeline(config, admin_path=tmp_path / "missing.toml").run(request)
    message = str(caught.value)
    assert "detector" in message
    assert "swapper" in message
    assert guard_commercial(config) is None


def test_tensorrt_options_and_cpu_fallback(tmp_path: Path) -> None:
    options = tensorrt_options(
        0,
        TensorRTConfig(
            engine_cache=tmp_path / "engines",
            timing_cache=tmp_path / "timing",
            fp16=True,
            builder_optimization_level=3,
        ),
    )
    assert options["trt_fp16_enable"] is True
    assert options["trt_builder_optimization_level"] == 3
    assert Path(str(options["trt_engine_cache_path"])).is_dir()
    assert Path(str(options["trt_timing_cache_path"])).is_dir()
    config = load_config(write_config(tmp_path, enabled=False))
    resolved = resolve_providers(["cpu"], 0, settings=config.execution)
    assert "CPUExecutionProvider" in resolved
    assert not (tmp_path / "engines-not-created").exists()


def test_run_feeds_uses_session_run_without_cuda() -> None:
    import barrett_faceswap.binding as binding

    binding._warned = False

    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, _outputs, feeds):
            self.calls += 1
            return [feeds["x"]]

    session = Session()
    value = np.zeros((1, 3), np.float32)
    outputs = run_feeds(session, {"x": value}, cuda_graph=True)
    assert session.calls == 1
    assert outputs[0] is value


def test_fp16_sibling_falls_back_and_is_selected_when_present(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=False))
    fp16 = config.model_copy(update={"execution": config.execution.model_copy(update={"precision": "fp16"})})
    assert weight_path(fp16, "swapper").name == "inswapper_128.onnx"
    sibling = config.models.swapper.path.with_name("inswapper_128_fp16.onnx")
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(b"not-a-model")
    assert weight_path(fp16, "swapper") == sibling
    assert weight_path(config, "swapper").name == "inswapper_128.onnx"


def test_ssim_and_identity_cosine() -> None:
    image = np.full((32, 32, 3), 40, np.uint8)
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    other = image.copy()
    other[:8] = 220
    assert ssim(image, other) < 0.99
    vector = np.array([1.0, 0.0, 0.0], np.float32)
    assert identity_cosine(vector, vector) == pytest.approx(1.0)
    assert identity_cosine(vector, np.array([0.0, 1.0, 0.0], np.float32)) == pytest.approx(0.0)


def test_dataset_scaffold_refuses_a_noncommercial_manifest(tmp_path: Path) -> None:
    good = REPO / "configs" / "dataset.example.yaml"
    spec = validate_dataset(good)
    plan = training_plan(spec)
    assert spec.samples
    assert len(plan) >= 4
    assert "512" in export_contract()
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nresolution: 256\nlicense: research-only\nsamples: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        validate_dataset(bad)
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text("name: x\nresolution: 64\nlicense: commercial-ok\nsamples: []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        validate_dataset(tiny)


def test_chinese_watermark_glyphs_leave_the_center_dark() -> None:
    frame = np.full((120, 160, 3), 180, np.uint8)
    chinese = apply_visible_watermark(frame, "人工智能生成")
    latin = apply_visible_watermark(frame, "AI-generated")
    assert not np.array_equal(chinese, latin)
    assert int(chinese.max()) > 200
    assert float(chinese[-4, 160 // 2].mean()) < 80


def test_disk_resume_does_not_watermark_twice(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, enabled=True))
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    frame = tmp_path / "frame_000001.png"
    write_png(source, (10, 10, 10), (80, 60))
    write_png(target, (20, 20, 20), (80, 60))
    write_png(frame, (200, 180, 160), (160, 120))
    consent = tmp_path / "consent.json"
    write_consent(consent)
    policy = CompliancePolicy(config, admin_path=tmp_path / "missing.toml")
    context = policy.before_process(
        SwapRequest(
            source_path=source,
            target_path=target,
            output_path=tmp_path / "out.png",
            enable_compliance=True,
            consent_file=consent,
        )
    )
    assert watermark_frames(policy, context, [frame]) == 1
    marked = frame.read_bytes()
    assert watermark_frames(policy, context, [frame]) == 0
    assert frame.read_bytes() == marked
    assert Path(str(frame) + ".wm").is_file()


def test_live_benchmark_line_includes_stages(tmp_path: Path) -> None:
    from barrett_faceswap.benchmark import run_live_benchmark

    config = load_config(write_config(tmp_path, enabled=False))
    report = run_live_benchmark(config, frames=3, width=32, height=24, dry_run=True, providers=["cpu"])
    text = report.line()
    assert "realtime_fps=" in text
    assert "stages=" in text
    assert "capture:" in text
    assert report.fps > 0


def test_training_scripts_do_not_train() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    dataset = str(REPO / "configs" / "dataset.example.yaml")
    checked = subprocess.run(
        [sys.executable, "scripts/train_swapper.py", "--dataset", dataset, "--check"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0
    refused = subprocess.run(
        [sys.executable, "scripts/train_swapper.py", "--dataset", dataset],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2
    exported = subprocess.run(
        [sys.executable, "scripts/export_onnx.py", "--check"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert exported.returncode == 0
    assert "512" in exported.stdout
    missing = subprocess.run(
        [sys.executable, "scripts/export_onnx.py"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    scored = subprocess.run(
        [sys.executable, "scripts/eval_swapper.py", "--self-test"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert scored.returncode == 0


def test_cli_reference_matches_the_parser() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["COLUMNS"] = "100"
    completed = subprocess.run(
        [sys.executable, "scripts/build_cli_reference.py", "--check"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
