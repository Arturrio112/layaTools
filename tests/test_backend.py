import sys
import types

from layatools import backend


class FakeRouter:
    instances: list = []

    def __init__(self, **kwargs):
        self.calls = []
        FakeRouter.instances.append(self)

    def predict(self, state, questions, model=None):
        self.calls.append(model)
        return {"answers": {}, "routing": {"model": model or "english"}}


def _backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", types.SimpleNamespace(Router=FakeRouter))
    monkeypatch.setenv("LAYATOOLS_DEVICE", "cpu")
    FakeRouter.instances.clear()
    return backend.LayaBackend()


def test_default_leaves_routing_to_laya(monkeypatch):
    monkeypatch.delenv("LAYATOOLS_LAYA_MODEL", raising=False)
    b = _backend(monkeypatch)
    b.predict({"m": "hi"}, {})
    assert FakeRouter.instances[0].calls == [None]
    assert b.last_routing == {"model": "english"}


def test_auto_means_default(monkeypatch):
    monkeypatch.setenv("LAYATOOLS_LAYA_MODEL", " Auto ")
    assert backend.configured_model() is None


def test_env_selects_typed_decisions(monkeypatch):
    monkeypatch.setenv("LAYATOOLS_LAYA_MODEL", "Typed-Decisions")
    b = _backend(monkeypatch)
    b.predict({"m": "hi"}, {})
    assert FakeRouter.instances[0].calls == ["typed-decisions"]
    assert b.last_routing == {"model": "typed-decisions"}
