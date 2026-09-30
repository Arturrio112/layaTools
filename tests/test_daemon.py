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
