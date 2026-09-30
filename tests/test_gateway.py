from pathlib import Path

import pytest

from layatools.gateway import Gateway, UnknownProfile
from layatools.profiles import load_profiles


class FakeBackend:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions))
        return {
            "answers": {
                "department": {"choice": "billing", "confidence": 0.94},
                "urgent": {"noul": 0.4, "confidence": 0.5},
            }
        }


@pytest.fixture
def gateway():
    profiles = load_profiles(Path(__file__).parent, builtin=False)
    return Gateway(FakeBackend(), profiles)


def test_decide_is_compact_and_escalates_low_confidence(gateway):
    result = gateway.decide("support_route", "charged twice")
    assert result == {
        "answers": {
            "department": {"value": "billing", "conf": 0.94},
            "urgent": {"value": False, "conf": 0.5},
        },
        "escalate": ["urgent"],
    }


def test_plain_text_goes_under_profile_state_field(gateway):
    gateway.decide("support_route", "hello")
    assert gateway.backend.calls[0][0] == {"message": "hello"}


def test_min_confidence_override(gateway):
    assert gateway.decide("support_route", "x", min_confidence=0.1)["escalate"] == []


def test_unknown_profile(gateway):
    with pytest.raises(UnknownProfile):
        gateway.decide("nope", "x")


def test_list_decisions(gateway):
    (item,) = gateway.list_decisions()
    assert item["answers"]["department"] == "choice: billing|technical|other"
    assert item["answers"]["urgent"] == "yes/no"


def test_rank_orders_by_score():
    class Rank:
        def predict(self, state, questions, model=None):
            hit = "auth" in state["content"]
            return {"answers": {"relevance": {"score": 2.5 if hit else 0.2},
                                "relevant": {"noul": 0.9 if hit else 0.1}}}

    g = Gateway(Rank(), load_profiles())
    out = g.rank("fix login", {"a.py": "unrelated", "auth.py": "auth code"})
    assert [r["id"] for r in out] == ["auth.py", "a.py"]


def test_shipped_profiles_are_generic_only():
    assert "relevance" in load_profiles()
    assert not {"tool_or_skill", "watchdog"} & set(load_profiles())


def test_project_profiles_override(tmp_path):
    from layatools.profiles import PROJECT_SUBDIR, profiles_for

    d = tmp_path / PROJECT_SUBDIR
    d.mkdir(parents=True)
    (d / "mine.yaml").write_text("description: x\nquestions:\n  q: {type: noul, instructions: hi}\n")
    assert "mine" in profiles_for(tmp_path)


def test_rank_unknown_profile(gateway):
    with pytest.raises(UnknownProfile):
        gateway.rank("x", {"a": "b"}, profile="nope")


def test_env_profile_dirs(tmp_path, monkeypatch):
    import os

    from layatools.profiles import PROJECT_SUBDIR, profiles_for

    a, b = tmp_path / "a", tmp_path / "b"
    for d, desc in ((a, "from a"), (b, "from b")):
        d.mkdir()
        (d / "shared.yaml").write_text(f"description: {desc}\nquestions:\n  q: {{type: noul, instructions: hi}}\n")
    (a / "only_a.yaml").write_text("questions:\n  q: {type: noul, instructions: hi}\n")
    monkeypatch.setenv("LAYATOOLS_PROFILES", os.pathsep.join([str(a), "", str(b)]))
    got = profiles_for(None)
    assert "only_a" in got and got["shared"].description == "from b"  # later directory wins

    proj = tmp_path / "proj"
    (proj / PROJECT_SUBDIR).mkdir(parents=True)
    (proj / PROJECT_SUBDIR / "shared.yaml").write_text("description: project\nquestions:\n  q: {type: noul, instructions: hi}\n")
    assert profiles_for(proj)["shared"].description == "project"  # project beats the env dirs
