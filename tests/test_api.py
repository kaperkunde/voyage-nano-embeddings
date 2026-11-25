from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import ConfigError, Settings
from tests.conftest import KEY, FakeEncoder, make_settings


def post(client: TestClient, auth: dict[str, str], body: object):
    return client.post("/v1/embeddings", json=body, headers=auth)


# --- shape ---------------------------------------------------------------


def test_openai_shape(client, auth):
    r = post(client, auth, {"input": ["a b", "c"], "model": "voyage-4-nano"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "MongoDB/voyage-4-nano"
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert all(d["object"] == "embedding" for d in body["data"])
    assert len(body["data"][0]["embedding"]) == 1024
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"] > 0


def test_single_string_input(client, auth):
    r = post(client, auth, {"input": "hello"})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1


def test_model_is_optional_and_checked(client, auth):
    assert post(client, auth, {"input": "x", "model": "MongoDB/voyage-4-nano"}).status_code == 200
    r = post(client, auth, {"input": "x", "model": "text-embedding-3-small"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_base64_is_little_endian_float32(client, auth):
    plain = post(client, auth, {"input": "x", "dimensions": 256}).json()["data"][0]["embedding"]
    packed = post(client, auth, {"input": "x", "dimensions": 256, "encoding_format": "base64"})
    raw = base64.b64decode(packed.json()["data"][0]["embedding"])
    assert np.allclose(np.frombuffer(raw, dtype="<f4"), plain, atol=1e-6)


# --- input_type routing ---------------------------------------------------


def test_input_type_routing(client, auth, encoder):
    post(client, auth, {"input": "q", "input_type": "query"})
    post(client, auth, {"input": "d", "input_type": "document"})
    post(client, auth, {"input": "n", "input_type": None})
    post(client, auth, {"input": "o"})
    assert [kind for _, kind in encoder.calls] == ["query", "document", "document", "document"]


def test_query_and_document_differ(client, auth):
    q = post(client, auth, {"input": "same", "input_type": "query"}).json()["data"][0]["embedding"]
    d = post(client, auth, {"input": "same"}).json()["data"][0]["embedding"]
    assert q != d


def test_bad_input_type(client, auth):
    r = post(client, auth, {"input": "x", "input_type": "passage"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_input_type"


# --- dimensions -------------------------------------------------------------


@pytest.mark.parametrize("dimensions", [256, 512, 1024, 2048])
def test_dimensions_are_normalised(client, auth, dimensions):
    r = post(client, auth, {"input": ["a", "b"], "dimensions": dimensions})
    for item in r.json()["data"]:
        vector = np.array(item["embedding"])
        assert len(vector) == dimensions
        assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


def test_truncation_keeps_the_prefix_direction(client, auth):
    full = np.array(
        post(client, auth, {"input": "x", "dimensions": 2048}).json()["data"][0]["embedding"]
    )
    cut = np.array(
        post(client, auth, {"input": "x", "dimensions": 512}).json()["data"][0]["embedding"]
    )
    expected = full[:512] / np.linalg.norm(full[:512])
    assert np.allclose(cut, expected, atol=1e-5)


@pytest.mark.parametrize("dimensions", [100, 0, -1, "1024", 1024.5, True])
def test_bad_dimensions(client, auth, dimensions):
    r = post(client, auth, {"input": "x", "dimensions": dimensions})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_dimensions"


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({}, "missing_input"),
        ({"input": []}, "invalid_input"),
        ({"input": 5}, "invalid_input"),
        ({"input": [1, 2]}, "invalid_input"),
        ({"input": [[1, 2]]}, "invalid_input"),
        ({"input": ""}, "empty_input"),
        ({"input": ["ok", "   "]}, "empty_input"),
        ({"input": "x", "encoding_format": "int8"}, "invalid_encoding_format"),
        ([], "invalid_body"),
    ],
)
def test_invalid_requests(client, auth, body, code):
    r = post(client, auth, body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == code
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}


def test_invalid_json(client, auth):
    r = client.post("/v1/embeddings", content=b"{nope", headers=auth)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_json"


def test_unknown_fields_are_ignored(client, auth):
    assert post(client, auth, {"input": "x", "user": "someone", "extra": 1}).status_code == 200


# --- limits -------------------------------------------------------------------------


def limited(**env: str) -> tuple[TestClient, FakeEncoder]:
    encoder = FakeEncoder()
    return TestClient(create_app(encoder, make_settings(**env))), encoder


def test_max_inputs(auth):
    client, encoder = limited(MAX_INPUTS="2")
    assert post(client, auth, {"input": ["a", "b"]}).status_code == 200
    r = post(client, auth, {"input": ["a", "b", "c"]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "too_many_inputs"
    assert len(encoder.calls) == 1


def test_max_input_chars(auth):
    client, _ = limited(MAX_INPUT_CHARS="10")
    assert post(client, auth, {"input": "x" * 10}).status_code == 200
    r = post(client, auth, {"input": ["ok", "x" * 11]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "input_too_long"
    assert "input[1]" in r.json()["error"]["message"]


def test_max_total_chars(auth):
    client, _ = limited(MAX_INPUT_CHARS="10", MAX_TOTAL_CHARS="25")
    assert post(client, auth, {"input": ["x" * 10, "y" * 10]}).status_code == 200
    r = post(client, auth, {"input": ["x" * 10] * 3})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "request_too_long"


def test_body_size_cap(auth):
    client, encoder = limited(MAX_TOTAL_CHARS="100")
    r = client.post("/v1/embeddings", content=b" " * (100 * 6 + 65536 + 1), headers=auth)
    assert r.status_code == 413
    assert encoder.calls == []


def test_validation_runs_before_the_model(client, auth, encoder):
    post(client, auth, {"input": "x", "dimensions": 7})
    assert encoder.calls == []


def test_too_many_waiting_is_429(auth):
    client, _ = limited(MAX_PENDING="1")
    # Pretend one request is running: the next is refused, not queued.
    client.app.state.pending = 1
    r = post(client, auth, {"input": "x"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "5"
    assert r.json()["error"]["type"] == "rate_limit_error"


# --- auth ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": KEY},
        {"Authorization": f"Basic {KEY}"},
        {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {KEY}x"},
    ],
)
def test_embeddings_need_the_key(client, headers):
    r = client.post("/v1/embeddings", json={"input": "x"}, headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_auth_comes_before_validation(client):
    r = client.post("/v1/embeddings", content=b"{nope")
    assert r.status_code == 401


def test_bearer_scheme_is_case_insensitive(client):
    r = client.post(
        "/v1/embeddings", json={"input": "x"}, headers={"Authorization": f"bearer {KEY}"}
    )
    assert r.status_code == 200


def test_health_and_openapi_need_no_key(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "MongoDB/voyage-4-nano",
        "revision": "abc123",
        "dim": 1024,
        "dimensions": [256, 512, 1024, 2048],
    }
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert list(spec.json()["paths"]) == ["/v1/embeddings"]
    assert spec.json()["security"] == [{"bearerAuth": []}]


def test_no_docs_pages_and_json_404(client, auth):
    for path in ("/docs", "/redoc", "/"):
        r = client.get(path, headers=auth)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"


# --- settings ---------------------------------------------------------------------------


def test_refuses_to_start_without_a_key():
    for env in ({}, {"API_KEY": ""}, {"API_KEY": "   "}):
        with pytest.raises(ConfigError, match="API_KEY"):
            Settings.from_env(env)


def test_settings_defaults_and_bounds():
    settings = Settings.from_env({"API_KEY": "k"})
    assert settings.threads >= 1
    assert (settings.max_inputs, settings.max_input_chars) == (64, 32000)
    with pytest.raises(ConfigError, match="THREADS"):
        Settings.from_env({"API_KEY": "k", "THREADS": "0"})
    with pytest.raises(ConfigError, match="MAX_INPUTS"):
        Settings.from_env({"API_KEY": "k", "MAX_INPUTS": "many"})
    with pytest.raises(ConfigError, match="DTYPE"):
        Settings.from_env({"API_KEY": "k", "DTYPE": "float16"})
