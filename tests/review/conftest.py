import os, pytest
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("BARRETT_DEV_MODE", raising=False)
    monkeypatch.delenv("BARRETT_COMPLIANCE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
