from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.config import NATIVE_DIMENSIONS, Settings

KEY = "test-key-0123456789abcdef"


class FakeEncoder:
    """A deterministic stand-in: a vector from the text's hash, and a different
    one for a query and a document, so routing is visible. No model, no torch."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def encode(self, texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
        self.calls.append((list(texts), input_type))
        rows = []
        for text in texts:
            digest = hashlib.sha256(f"{input_type}:{text}".encode()).digest()
            seed = int.from_bytes(digest[:8], "big")
            row = np.random.default_rng(seed).normal(size=NATIVE_DIMENSIONS).astype(np.float32)
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows), sum(len(t.split()) + 1 for t in texts)


def make_settings(**overrides: str) -> Settings:
    env = {"API_KEY": KEY, "THREADS": "2", "MODEL_REVISION": "abc123"}
    env.update(overrides)
    return Settings.from_env(env)


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def client(encoder: FakeEncoder) -> TestClient:
    return TestClient(create_app(encoder, make_settings()))


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {KEY}"}
