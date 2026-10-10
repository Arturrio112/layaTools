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


def test_baseline_agreement_and_log_option(log_root, capsys):
    own = log_root / "own.jsonl"
    ans = lambda v: {"answers": {"q": {"value": v, "conf": 0.9}}}  # noqa: E731
    a = decision_log.log_decision("p", "t1", ans("x"), path=own, baseline={"q": "x"})
    decision_log.log_decision("p", "t2", ans("x"), path=own, baseline={"q": {"value": "y"}})
    decision_log.log_decision("p", "t3", ans("x"), path=own)  # no baseline: not counted
    decision_log.log_outcome(a, {"labels": {"q": "x"}}, own)
    out = calibrate.baseline_agreement(decision_log.read(own))
    assert out == {"q": {"n": 2, "agreement": 0.5, "labelled": 1, "laya_accuracy": 1.0, "baseline_accuracy": 1.0}}
    assert decision_log.read() == []

def test_cli_eval_and_export_read_a_given_log(log_root, monkeypatch, capsys):
    own = log_root / "own.jsonl"
    a = decision_log.log_decision("p", "t1", {"answers": {"q": {"value": "x", "conf": 0.9}}}, path=own, baseline={"q": "z"})
    decision_log.log_decision("p", "hidden", {"answers": {}}, path=own, redact=True)
    decision_log.log_outcome(a, {"labels": {"q": "x"}}, own)
    monkeypatch.setattr(cli.daemon, "call", lambda path, payload: [{"answers": {"q": {"value": "x", "conf": 0.8}}}])
    cli.main(["eval", "p", "--log", "own.jsonl"])
    out = json.loads(capsys.readouterr().out)
    assert out["q"]["accuracy"] == 1.0 and out["baseline_agreement"]["q"]["agreement"] == 0.0
    cli.main(["export", "--log", "own.jsonl", "--with-outcome"])
    assert [json.loads(line)["id"] for line in capsys.readouterr().out.splitlines()] == [a]
    cli.main(["outcome", a, '{"passed": true}', "--log", "own.jsonl"])
    assert decision_log.read(own)[0]["outcome"] == {"passed": True}


def test_cli_log_args_are_confined(log_root, tmp_path, capsys):
    import pytest

    for argv in (["export", "--log", str(tmp_path / "x.jsonl")], ["eval", "p", "--log", "../x.jsonl"],
                 ["outcome", "i", "{}", "--log", "/etc/x"], ["prune", "--log", "../x.jsonl"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert str(log_root) in capsys.readouterr().err


DAY = 86400


def _seed(path, now):
    """old unlabelled, old labelled (+2 outcomes), old with unlabelled outcome, recent; returns ids."""
    ids = {}
    for name, age in (("old", 100), ("old_labelled", 120), ("old_outcome", 110), ("recent", 1)):
        ids[name] = decision_log.log_decision("p", name * 50, {"answers": {}}, path=path)
    # rewrite timestamps (ts is set at write time)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    ages = {ids["old"]: 100, ids["old_labelled"]: 120, ids["old_outcome"]: 110, ids["recent"]: 1}
    for rec in lines:
        rec["ts"] = now - ages[rec["id"]] * DAY
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    decision_log.log_outcome(ids["old_labelled"], {"labels": {"q": "x"}}, path)
    decision_log.log_outcome(ids["old_outcome"], {"passed": True}, path)
    return ids


def test_prune_keeps_recent_and_labelled(log_root):
    import time

    now = time.time()
    path = log_root / "a.jsonl"
    ids = _seed(path, now)
    before = path.read_bytes()
    dry = decision_log.prune(path, 90 * DAY, dry_run=True, now=now)
    assert path.read_bytes() == before and dry["after"]["decisions"] == 2 and dry["before"]["decisions"] == 4
    out = decision_log.prune(path, 90 * DAY, now=now)
    assert {r["id"] for r in decision_log.read(path)} == {ids["old_labelled"], ids["recent"]}
    assert decision_log.read(path)[0]["outcome"] == {"labels": {"q": "x"}}
    assert out["after"]["outcomes"] == 1 and out["after"]["bytes"] == path.stat().st_size < out["before"]["bytes"]
    assert not list(log_root.glob("*.tmp*"))  # atomic rewrite leaves no temp file
    decision_log.prune(path, 90 * DAY, keep_labelled=False, now=now)
    assert [r["id"] for r in decision_log.read(path)] == [ids["recent"]]


def test_prune_max_mb_trims_oldest_unlabelled(log_root):
    import time

    now = time.time()
    path = log_root / "b.jsonl"
    ids = _seed(path, now)
    one = len(json.dumps(decision_log.read(path)[3]))  # about one recent record
    limit = path.stat().st_size - 1
    out = decision_log.prune(path, 1000 * DAY, max_bytes=limit, now=now)  # age alone removes nothing
    kept = [r["id"] for r in decision_log.read(path)]
    assert ids["old"] not in kept and ids["old_labelled"] in kept and ids["recent"] in kept
    assert out["after"]["bytes"] <= limit and one > 0


def test_cli_prune_and_auto_prune(log_root, monkeypatch, capsys):
    import time

    now = time.time()
    sub = log_root / "factory" / "c.jsonl"
    sub.parent.mkdir()
    _seed(sub, now)
    glob = decision_log.log_path()
    _seed(glob, now)
    cli.main(["prune", "--log", "factory/c.jsonl", "--older-than", "90d", "--dry-run"])
    rep = json.loads(capsys.readouterr().out)
    assert rep[0]["dry_run"] and rep[0]["before"]["decisions"] == 4 and len(decision_log.read(sub)) == 4
    monkeypatch.delenv("LAYATOOLS_PRUNE_DAYS", raising=False)
    assert decision_log.auto_prune() == []  # unset = off
    monkeypatch.setenv("LAYATOOLS_PRUNE_DAYS", "90")
    done = decision_log.auto_prune(now=now)
    assert [str(r["path"]) for r in done] == [str(sub)] and len(decision_log.read(sub)) == 2
    assert len(decision_log.read(glob)) == 4  # the global log is only pruned when opted in
    monkeypatch.setenv("LAYATOOLS_PRUNE_GLOBAL", "1")
    decision_log.auto_prune(now=now)
    assert len(decision_log.read(glob)) == 2
    cli.main(["prune", "--all", "--older-than", "10d", "--no-keep-labelled"])
    assert len(decision_log.read(glob)) == 1
    with __import__("pytest").raises(SystemExit):
        cli.main(["prune"])  # needs --log or --all
