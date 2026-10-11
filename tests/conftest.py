import pytest


@pytest.fixture(autouse=True)
def decision_log_file(tmp_path, monkeypatch):
    """Every test logs decisions to its own file, never to the real ~/.local/share log."""
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("LAYATOOLS_DECISION_LOG", str(path))
    return path


@pytest.fixture(autouse=True)
def log_root(tmp_path, monkeypatch):
    """Per-request log files may only live under this root."""
    root = tmp_path / "logroot"
    root.mkdir()
    monkeypatch.setenv("LAYATOOLS_LOG_ROOT", str(root))
    return root.resolve()
