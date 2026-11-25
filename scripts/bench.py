"""Latency of a running server, per request shape, after a warm-up.

API_KEY=... python scripts/bench.py http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
KEY = os.environ["API_KEY"]
SENTENCE = "Self-hosted embeddings let a search find text by meaning instead of by words. "


def run(body: dict) -> float:
    request = urllib.request.Request(  # noqa: S310
        BASE + "/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    started = time.perf_counter()
    urllib.request.urlopen(request, timeout=600).read()  # noqa: S310
    return time.perf_counter() - started


def measure(label: str, body: dict, repeats: int) -> None:
    run(body)  # warm-up: the first call of a new shape pays for its kernels
    times = sorted(run(body) for _ in range(repeats))
    print(
        f"{label:<42} median {statistics.median(times):6.2f}s  "
        f"min {times[0]:6.2f}s  max {times[-1]:6.2f}s  (n={repeats})"
    )


measure(
    "1 query (about 12 tokens)",
    {"input": "Which planet is the Red Planet?", "input_type": "query"},
    15,
)
measure("1 document of about 200 tokens", {"input": SENTENCE * 14}, 8)
measure("batch of 50 short sentences", {"input": [SENTENCE] * 50}, 4)
measure("batch of 20 documents of about 200 tokens", {"input": [SENTENCE * 14] * 20}, 3)
measure("1 document of about 2000 tokens", {"input": SENTENCE * 140}, 3)
