import pytest


@pytest.fixture(autouse=True)
def decision_log_file(tmp_path, monkeypatch):
    """Every test logs decisions to its own file, never to the real ~/.local/share log."""
    path = tmp_path / "decisions.jsonl"
    monkeypatch.setenv("LAYATOOLS_DECISION_LOG", str(path))
    return path
