import re, shutil, subprocess
from pathlib import Path
from unittest import mock
import numpy as np, pytest, yaml

REPO = Path(__file__).resolve().parents[2]
from barrett_faceswap import config as cfg
from barrett_faceswap.types import Face, SwapRequest

# ---------- compliance toggles ----------
def _cfg(tmp_path, local=None):
    d = tmp_path / "configs"; d.mkdir()
    shutil.copy(REPO / "configs" / "default.yaml", d / "default.yaml")
    if local is not None:
        (d / "local.yaml").write_text(yaml.safe_dump(local))
    return d / "default.yaml"

def test_compliance_default_on(tmp_path):
    assert cfg.load_config(_cfg(tmp_path)).compliance.enabled is True
    assert cfg.ComplianceConfig().enabled is True

@pytest.mark.parametrize("env,val,exp", [
    ("BARRETT_DEV_MODE", "1", False), ("BARRETT_DEV_MODE", "0", True),
    ("BARRETT_DEV_MODE", "", True), ("BARRETT_DEV_MODE", "garbage", True),
    ("BARRETT_COMPLIANCE", "off", False), ("BARRETT_COMPLIANCE", "on", True),
    ("BARRETT_COMPLIANCE", "maybe", True)])
def test_compliance_env(tmp_path, monkeypatch, env, val, exp):
    monkeypatch.setenv(env, val)
    assert cfg.load_config(_cfg(tmp_path)).compliance.enabled is exp

def test_explicit_on_beats_dev_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("BARRETT_DEV_MODE", "1"); monkeypatch.setenv("BARRETT_COMPLIANCE", "on")
    assert cfg.load_config(_cfg(tmp_path)).compliance.enabled is True

def test_local_yaml_optout_and_env_override(tmp_path, monkeypatch):
    p = _cfg(tmp_path, {"compliance": {"enabled": False}})
    assert cfg.load_config(p).compliance.enabled is False
    monkeypatch.setenv("BARRETT_COMPLIANCE", "on")
    assert cfg.load_config(p).compliance.enabled is True

def test_local_yaml_partial_keeps_other_keys(tmp_path):
    p = _cfg(tmp_path, {"compliance": {"watermark": {}}})
    assert cfg.load_config(p).compliance.enabled is True

def test_repo_has_no_committed_local_yaml():
    assert not (REPO / "configs" / "local.yaml").exists()

# ---------- i18n ----------
def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(_flat(v, p + k + "."))
        else: out[p + k] = v
    return out

def test_i18n_keys_and_placeholders_match():
    loc = REPO / "src/barrett_faceswap/i18n/locales"
    en = _flat(yaml.safe_load((loc / "en.yaml").read_text()))
    zh = _flat(yaml.safe_load((loc / "zh_CN.yaml").read_text()))
    assert set(en) == set(zh), (set(en) ^ set(zh))
    ph = re.compile(r"\{(\w+)\}")
    bad = [k for k in en if set(ph.findall(str(en[k]))) != set(ph.findall(str(zh[k])))]
    assert not bad

def test_i18n_literal_keys_resolve():
    loc = REPO / "src/barrett_faceswap/i18n/locales"
    en = _flat(yaml.safe_load((loc / "en.yaml").read_text()))
    used = set()
    for f in (REPO / "src").rglob("*.py"):
        used |= set(re.findall(r"\bt\(\s*[\"']([\w.]+)[\"']", f.read_text()))
    assert not sorted(k for k in used if k not in en)

# ---------- face library ----------
from barrett_faceswap.library import FaceLibrary

def test_library_create_list_delete(tmp_path):
    lib = FaceLibrary(tmp_path / "lib")
    s = lib.create("Alice")
    assert [p.name for p in lib.list_profiles()] == ["Alice"]
    lib.delete(s.id if hasattr(s, "id") else "Alice")
    assert lib.list_profiles() == []

def test_library_create_traversal_name_stays_inside(tmp_path):
    lib = FaceLibrary(tmp_path / "lib")
    lib.create("../../evil")
    assert not (tmp_path / "evil").exists()
    assert all(p.parent == tmp_path / "lib" for p in (tmp_path / "lib").iterdir())

def test_library_delete_relative_traversal(tmp_path):
    lib = FaceLibrary(tmp_path / "lib"); lib.create("x")
    victim = tmp_path / "victim"; victim.mkdir(); (victim / "profile.json").write_text("{}")
    try: lib.delete("../victim")
    except Exception: pass
    assert victim.exists(), "delete('../victim') removed a dir outside the library root"

def test_library_delete_absolute_path(tmp_path):
    lib = FaceLibrary(tmp_path / "lib"); lib.create("x")
    victim = tmp_path / "abs"; victim.mkdir(); (victim / "profile.json").write_text("{}")
    try: lib.delete(str(victim))
    except Exception: pass
    assert victim.exists()

# ---------- recovery / empty shots ----------
from barrett_faceswap.recover import FaceStream
def _face():
    return Face(bbox=np.array([10, 10, 50, 50], float), score=0.9,
                landmarks=np.array([[20, 20], [40, 20], [30, 30], [22, 40], [38, 40]], float))
FR = np.zeros((64, 64, 3), np.uint8)

def test_all_empty_frames_pass_through_untouched():
    s = FaceStream(max_gap=3); out = []
    for i in range(5):
        f = FR.copy(); out += s.push(i, f, [], "miss")
    out += s.finish()
    assert [r.tag for r in out] == ["passthrough"] * 5
    assert all(r.faces == [] and np.array_equal(r.frame, FR) for r in out)

def test_long_gap_beyond_max_gap_passes_through():
    s = FaceStream(max_gap=1); out = s.push(0, FR, [_face()], "detected")
    for i in range(1, 6): out += s.push(i, FR, [], "miss")
    out += s.finish()
    assert [r.tag for r in out][2:] == ["passthrough"] * 4

def test_empty_tail_after_face_not_fabricated():
    # A real empty shot (scene cut, no face) right after a face shot gets a copied face.
    s = FaceStream(max_gap=3); out = s.push(0, FR, [_face()], "detected")
    out += s.push(1, FR, [], "miss"); out += s.finish()
    assert out[-1].tag == "passthrough" and out[-1].faces == []

# ---------- output overwrite / pipe cleanup ----------
from barrett_faceswap.compliance import watermark as wm
def test_save_image_does_not_overwrite(tmp_path):
    p = tmp_path / "o.png"; p.write_bytes(b"ORIGINAL")
    try: wm.save_image(np.zeros((8, 8, 3), np.uint8), p, watermark=False, text="x")
    except FileExistsError: pass
    assert p.read_bytes() == b"ORIGINAL"

from barrett_faceswap.media import ffmpeg as ff
def test_encode_commands_have_no_force_overwrite(tmp_path):
    import inspect
    src = inspect.getsource(ff.build_encode_command) + inspect.getsource(ff.build_encode_pipe_command)
    assert '"-y"' not in src and "'-y'" not in src

def test_pipe_failure_keeps_preexisting_output(tmp_path):
    from barrett_faceswap.video import VideoPipeline
    out = tmp_path / "existing.mp4"; out.write_bytes(b"KEEP")
    vp = VideoPipeline.__new__(VideoPipeline); vp.config = mock.MagicMock(); vp.policy = mock.MagicMock()
    vp._on_progress = None
    req = SwapRequest(source_path=tmp_path / "s.mp4", target_path=tmp_path / "t.mp4", output_path=out)
    with pytest.raises(Exception):
        vp._run_pipe(req, None, None)  # source is a video -> _check_paths raises
    assert out.exists()

def test_rawvideopipe_kills_decoder_if_encoder_fails(tmp_path):
    dec = mock.MagicMock()
    with mock.patch.object(ff.subprocess, "Popen", side_effect=[dec, OSError("boom")]):
        with pytest.raises(OSError):
            ff.RawVideoPipe(tmp_path / "s.mp4", tmp_path / "o.mp4", width=16, height=16, fps=25.0,
                            encoder="libx264", crf=20, audio_source=None, audio_mode="keep",
                            trim_start=None, trim_end=None, seek=None)
    assert dec.kill.called or dec.terminate.called
