import json
import stat

from layatools import calibrate, cli, decision_log
from layatools.server import search_code


def test_log_and_join_outcomes(decision_log_file):
    a = decision_log.log_decision("p", "text a", {"answers": {"q": {"value": "x", "conf": 0.9}}, "escalate": []}, {"t": 1})
    b = decision_log.log_decision("other", {"message": "b"}, {"answers": {}, "escalate": []})
    decision_log.log_outcome(a, {"labels": {"q": "y"}})
    decision_log.log_outcome(a, {"labels": {"q": "x"}})  # the latest outcome wins
    with decision_log_file.open("a") as f:
        f.write('{"type": "decision", "id": "torn')  # a crash mid-write must not hide the rest
    c = decision_log.log_decision("p", "after the tear", {"answers": {}})  # must not be glued onto the torn line
    rows = decision_log.read()
    assert [r["id"] for r in rows] == [a, b, c]
    assert rows[0]["outcome"] == {"labels": {"q": "x"}} and rows[0]["meta"] == {"t": 1}
    assert rows[1]["outcome"] is None
    assert [r["id"] for r in decision_log.read(profile="p")] == [a, c]


def test_logging_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYATOOLS_DECISION_LOG", "off")
    assert decision_log.log_decision("p", "x", {"answers": {}}) is None
    assert not decision_log.log_outcome("id", {"a": 1})
    assert decision_log.read() == []


def test_long_state_is_clipped(decision_log_file):
    decision_log.log_decision("p", {"ticket": "x" * 10000}, {"answers": {}})
    assert len(decision_log.read()[0]["state"]["ticket"]) == decision_log.MAX_FIELD_CHARS


def test_correct_by_answer_type():
    assert calibrate.correct("Billing", "billing")
    assert calibrate.correct(True, "yes") and not calibrate.correct(False, "yes")
    assert calibrate.correct(2.4, 2) and not calibrate.correct(2.6, 2)


def test_calibrate_picks_lowest_threshold_reaching_target():
    # Confident answers are right, unsure ones are coin flips.
    pairs = [("a", 0.95, "a")] * 6 + [("a", 0.55, "a"), ("a", 0.5, "b"), ("a", 0.45, "b"), ("a", 0.4, "a")]
    out = calibrate.calibrate(pairs, target=0.9)
    assert out["n"] == 10 and out["accuracy"] == 0.8
    assert out["recommended_min_confidence"] == 0.55  # 7/7 right from 0.55 up; 7/8 at 0.5 is below target
    assert out["coverage_at_recommended"] == 0.7
    assert calibrate.calibrate([("a", 0.9, "b")], target=0.9)["recommended_min_confidence"] is None


def test_evaluate_groups_by_question():
    results = [{"answers": {"q": {"value": "a", "conf": 0.9}, "r": {"value": True, "conf": 0.8}}}]
    report = calibrate.evaluate(results, [{"q": "a"}], target=0.9)
    assert set(report) == {"q"} and report["q"]["accuracy"] == 1.0


def test_cli_eval_from_logged_outcomes(monkeypatch, capsys):
    a = decision_log.log_decision("route", "small change", {"answers": {}})
    decision_log.log_outcome(a, {"labels": {"tier": "small"}})
    decision_log.log_decision("route", "no outcome yet", {"answers": {}})
    sent = {}

    def fake_call(path, payload):
        sent.update(payload)
        return [{"answers": {"tier": {"value": "small", "conf": 0.8}}, "escalate": []}]

    monkeypatch.setattr(cli.daemon, "call", fake_call)
    cli.main(["eval", "route"])
    out = json.loads(capsys.readouterr().out)
    assert sent["items"] == ["small change"] and sent["log"] is False and sent["min_confidence"] == 0.0
    assert out["tier"]["accuracy"] == 1.0 and "table" not in out["tier"]


def test_cli_outcome_and_export(capsys):
    a = decision_log.log_decision("p", "x", {"answers": {}})
    decision_log.log_decision("p", "y", {"answers": {}})
    cli.main(["outcome", a, '{"passed": true}'])
    capsys.readouterr()
    cli.main(["export", "--with-outcome"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["id"] for r in rows] == [a] and rows[0]["outcome"] == {"passed": True}


def test_mcp_search_runs_lt(tmp_path, monkeypatch):
    fake = tmp_path / "lt"
    fake.write_text('#!/bin/sh\necho "[{\\"path\\": \\"$2\\", \\"args\\": \\"$*\\"}]"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    out = json.loads(search_code("where is x", 3, "/proj", lt=str(fake)))
    assert out[0]["path"] == "/proj" and "search where is x -k 3 --json" in out[0]["args"]
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert "not installed" in json.loads(search_code("x"))["error"]
    failing = tmp_path / "lt-fail"
    failing.write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
    failing.chmod(0o755)
    assert json.loads(search_code("x", lt=str(failing))) == {"error": "boom"}


def test_example_profiles_load():
    from pathlib import Path

    from layatools.profiles import load_profiles

    got = load_profiles(Path(__file__).parent.parent / "examples" / "profiles", builtin=False)
    assert set(got) == {"status_triage", "stall_triage"}
    assert got["status_triage"].summary()["answers"]["kind"] == "choice: decision|finished|error|waiting|routine"
    assert got["stall_triage"].state_field == "tail"
