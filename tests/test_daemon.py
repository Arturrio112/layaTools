import base64
import http.client
import json
import struct
import threading
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from layatools.daemon import local_host, make_handler
from layatools.gateway import Gateway
from layatools.profiles import PROJECT_SUBDIR, load_profiles


class FakeBackend:
    def predict(self, state, questions, model=None):
        if "task" in state:
            hit = "auth" in state["content"]
            return {"answers": {"relevance": {"score": 2.5 if hit else 0.2},
                                "relevant": {"noul": 0.9 if hit else 0.1}}}
        return {"answers": {"department": {"choice": "billing", "confidence": 0.94},
                            "urgent": {"noul": 0.4, "confidence": 0.5}}}


class FakeEmbedder:
    model_name = "fake-embed"

    def embed(self, texts, *, query=False):
        return [[float(len(t)), 1.0 if query else 0.0] for t in texts]


@pytest.fixture
def server():
    profiles = load_profiles(Path(__file__).parent)  # shipped `relevance` + tests/support_route.yaml
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Gateway(FakeBackend(), profiles), FakeEmbedder()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/json"}
    if host:
        headers["Host"] = host
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, data, headers)
    resp = conn.getresponse()
    out = resp.status, json.loads(resp.read())
    conn.close()
    return out


def test_health(server):
    assert request(server, "GET", "/health") == (200, {"ok": True, "embed_model": "fake-embed"})


def test_decide(server):
    code, out = request(server, "POST", "/v1/decide", {"profile": "support_route", "text": "charged twice"})
    assert code == 200
    assert out["answers"]["department"] == {"value": "billing", "conf": 0.94}
    assert out["escalate"] == ["urgent"]


def test_rank(server):
    code, out = request(server, "POST", "/v1/rank", {"task": "fix login", "items": {"a": "x", "b": "auth"}})
    assert code == 200
    assert [r["id"] for r in out] == ["b", "a"]


def test_embed_is_packed_float32(server):
    code, out = request(server, "POST", "/v1/embed", {"texts": ["ab", "abcd"], "query": True})
    assert code == 200 and out["model"] == "fake-embed" and (out["dim"], out["count"]) == (2, 2)
    assert list(struct.unpack("<4f", base64.b64decode(out["data"]))) == [2.0, 1.0, 4.0, 1.0]


def test_unknown_profile_and_bad_requests_are_400(server):
    assert request(server, "POST", "/v1/decide", {"profile": "nope", "text": "x"})[0] == 400
    assert request(server, "POST", "/v1/rank", {"task": "x", "items": {}, "profile": "nope"})[0] == 400
    assert request(server, "POST", "/v1/decide", {"text": "x"})[0] == 400
    assert request(server, "POST", "/v1/decide", b"not json")[0] == 400
    assert request(server, "POST", "/v1/decide", [1, 2])[0] == 400


def test_unknown_paths_are_404(server):
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/v1/nope", {})[0] == 404


def test_decisions_use_project_profiles(server, tmp_path):
    project = tmp_path / "my project"  # a space, to check the query string is decoded
    d = project / PROJECT_SUBDIR
    d.mkdir(parents=True)
    (d / "mine.yaml").write_text("description: x\nquestions:\n  q: {type: noul, instructions: hi}\n")
    code, out = request(server, "GET", "/v1/decisions?" + urllib.parse.urlencode({"cwd": str(project)}))
    assert code == 200 and "mine" in {p["name"] for p in out}
    code, out = request(server, "GET", "/v1/decisions")
    assert code == 200 and "mine" not in {p["name"] for p in out}


def test_foreign_host_is_rejected(server):
    assert request(server, "GET", "/health", host="evil.example:8765")[0] == 403
    assert request(server, "POST", "/v1/decide", {"profile": "support_route", "text": "x"}, host="evil.example")[0] == 403
    assert request(server, "GET", "/health", host=f"localhost:{server}")[0] == 200


def test_local_host():
    assert local_host(None) and local_host("127.0.0.1:8765") and local_host("LOCALHOST") and local_host("[::1]:1")
    assert not local_host("127.0.0.1.evil.com:8765") and not local_host("attacker.test")


def test_decide_is_logged_with_meta_and_id(server, decision_log_file):
    from layatools import decision_log

    code, out = request(server, "POST", "/v1/decide", {"profile": "support_route", "text": "x", "meta": {"ticket": "T1"}})
    assert code == 200 and len(out["id"]) == 32
    (row,) = decision_log.read()
    assert row["id"] == out["id"] and row["state"] == "x" and row["meta"] == {"ticket": "T1"}
    assert request(server, "POST", "/v1/outcome", {"id": out["id"], "outcome": {"passed": True}}) == (200, {"ok": True, "logged": True})
    assert decision_log.read()[0]["outcome"] == {"passed": True}
    assert request(server, "POST", "/v1/outcome", {"id": out["id"], "outcome": "nope"})[0] == 400


def test_decide_batch_and_no_log(server, decision_log_file):
    from layatools import decision_log

    code, out = request(server, "POST", "/v1/decide", {"profile": "support_route", "items": ["a", {"message": "b"}], "log": False})
    assert code == 200 and len(out) == 2 and all("id" not in r for r in out)
    assert decision_log.read() == []
    assert request(server, "POST", "/v1/decide", {"profile": "support_route", "items": "a"})[0] == 400


def test_log_path_redact_and_baseline(server, decision_log_file, tmp_path):
    import hashlib

    from layatools import decision_log

    own = tmp_path / "shadow.jsonl"
    body = {"profile": "support_route", "text": "client secret copy", "log_path": str(own), "redact": True,
            "meta": {"k": 1}, "baseline": {"department": {"value": "billing"}}}
    code, out = request(server, "POST", "/v1/decide", body)
    assert code == 200 and decision_log.read() == []  # the global log is untouched
    (row,) = decision_log.read(own)
    assert row["id"] == out["id"] and row["redacted"] and "state" not in row
    assert row["state_sha256"] == hashlib.sha256(b"client secret copy").hexdigest()
    assert "client secret" not in own.read_text()
    assert row["baseline"] == {"department": {"value": "billing"}} and row["answers"] and row["meta"] == {"k": 1}
    # outcomes go to the same file
    assert request(server, "POST", "/v1/outcome", {"id": out["id"], "outcome": {"ok": 1}, "log_path": str(own)})[0] == 200
    assert decision_log.read(own)[0]["outcome"] == {"ok": 1}


def test_log_path_is_validated(server, tmp_path):
    for bad in ("relative.jsonl", str(tmp_path / "nodir" / "x.jsonl"), 5):
        code, out = request(server, "POST", "/v1/decide", {"profile": "support_route", "text": "x", "log_path": bad})
        assert code == 400 and "log_path" in out["error"]


def test_batch_per_item_meta_and_baseline(server, decision_log_file):
    from layatools import decision_log

    code, out = request(server, "POST", "/v1/decide", {
        "profile": "support_route", "items": ["a", "b"], "meta": {"run": 1},
        "metas": [{"page": "home"}, None], "baselines": [{"department": "billing"}, {"department": "sales"}]})
    assert code == 200 and len(out) == 2
    a, b = decision_log.read()
    assert a["meta"] == {"run": 1, "page": "home"} and b["meta"] == {"run": 1}
    assert a["baseline"] == {"department": "billing"} and b["baseline"] == {"department": "sales"}
    bad = {"profile": "support_route", "items": ["a", "b"], "metas": [{}]}
    assert request(server, "POST", "/v1/decide", bad)[0] == 400


def test_rank_logging_is_opt_in(server, decision_log_file):
    from layatools import decision_log

    body = {"task": "fix login", "items": {"a": "x", "b": "auth"}}
    code, out = request(server, "POST", "/v1/rank", body)
    assert code == 200 and all("log_id" not in r for r in out) and decision_log.read() == []
    code, out = request(server, "POST", "/v1/rank", {**body, "log": True, "redact": True})
    rows = {r["id"]: r for r in decision_log.read()}
    assert [r["id"] for r in out] == ["b", "a"] and set(rows) == {r["log_id"] for r in out}
    top = rows[out[0]["log_id"]]
    assert top["meta"]["item"] == "b" and top["answers"]["relevance"]["value"] == out[0]["score"] and "state" not in top
