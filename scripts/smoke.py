"""Smoke test against a running server (the real image, with the real model).

    API_KEY=... python scripts/smoke.py http://127.0.0.1:8000

Standard library only. Checks health, the schema, auth, a retrieval ranking
that only a working model gets right, dimensions, base64, and the limits'
error shape; prints the time per call. Exits non-zero on the first failure.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ.get("API_KEY", "")


def call(path, body=None, key=KEY):
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data=data, headers=headers)  # noqa: S310
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            return response.status, json.load(response), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), time.perf_counter() - started


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        sys.exit(1)
    print(f"ok   {message}")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def main():
    check(KEY, "API_KEY is set in this shell")
    status, health, _ = call("/health", key=None)
    check(status == 200 and health["status"] == "ok", f"/health {health}")
    status, spec, _ = call("/openapi.json", key=None)
    check(status == 200 and "/v1/embeddings" in spec["paths"], "/openapi.json describes it")
    status, body, _ = call("/v1/embeddings", {"input": "x"}, key="wrong")
    check(status == 401, "a wrong key is refused with 401")
    status, body, _ = call("/v1/embeddings", {"input": "x", "dimensions": 7})
    check(status == 400 and body["error"]["param"] == "dimensions", "a bad dimension is a 400")

    docs = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        "Der Hund läuft schnell durch den Park, weil er seinen Ball sucht.",
    ]
    status, body, took = call("/v1/embeddings", {"input": docs, "input_type": "document"})
    check(status == 200 and len(body["data"]) == len(docs), f"documents embedded ({took:.2f}s)")
    vectors = [d["embedding"] for d in body["data"]]
    check(len(vectors[0]) == 1024, "default size is 1024")
    check(all(abs(math.sqrt(dot(v, v)) - 1) < 1e-3 for v in vectors), "vectors have length 1")

    status, body, took = call(
        "/v1/embeddings",
        {"input": "Which planet is known as the Red Planet?", "input_type": "query"},
    )
    check(status == 200, f"one query embedded ({took:.2f}s)")
    query = body["data"][0]["embedding"]
    scores = [dot(query, v) for v in vectors]
    best = max(range(len(docs)), key=scores.__getitem__)
    check(best == 1, f"the query finds the Mars sentence first (scores {scores[:4]})")

    status, german, _ = call(
        "/v1/embeddings", {"input": "Wo läuft der Hund hin?", "input_type": "query"}
    )
    gscores = [dot(german["data"][0]["embedding"], v) for v in vectors]
    check(
        max(range(len(docs)), key=gscores.__getitem__) == 4, "a German query finds the German text"
    )

    status, b64, _ = call(
        "/v1/embeddings",
        {
            "input": docs[1],
            "input_type": "document",
            "dimensions": 256,
            "encoding_format": "base64",
        },
    )
    raw = base64.b64decode(b64["data"][0]["embedding"])
    floats = struct.unpack("<256f", raw)
    check(status == 200 and abs(math.sqrt(dot(floats, floats)) - 1) < 1e-3, "base64, 256 dims")
    cosine = dot(
        floats, [x / math.sqrt(dot(vectors[1][:256], vectors[1][:256])) for x in vectors[1][:256]]
    )
    check(cosine > 0.999, f"256 dims is the renormalised start of the 1024 (cosine {cosine:.5f})")

    status, body, took = call("/v1/embeddings", {"input": ["a sentence about nothing much"] * 50})
    check(status == 200 and len(body["data"]) == 50, f"a batch of 50 short texts ({took:.2f}s)")
    status, body, took = call(
        "/v1/embeddings", {"input": ["word " * 200] * 20, "input_type": "document"}
    )
    check(status == 200, f"20 documents of about 200 tokens ({took:.2f}s)")
    print("smoke test passed")


main()
