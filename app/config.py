"""Settings, read once from the environment. Nothing here touches the network."""

from __future__ import annotations

import os
from dataclasses import dataclass

MODEL_ID = "MongoDB/voyage-4-nano"
# Names a request's `model` may use. Anything else is a clear 400, so a
# client configured for another model finds out at once.
MODEL_NAMES = ("voyage-4-nano", MODEL_ID)
# The model's Matryoshka sizes (its config.json `matryoshka_dimensions`).
DIMENSIONS = (256, 512, 1024, 2048)
DEFAULT_DIMENSIONS = 1024
NATIVE_DIMENSIONS = 2048
PROMPTS = {
    "query": "Represent the query for retrieving supporting documents: ",
    "document": "Represent the document for retrieval: ",
}


class ConfigError(Exception):
    """A setting that is missing or unusable: the server refuses to start."""


def _int(env: dict[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}.")
    return value


def available_cpus() -> int:
    """The cores this process may use: affinity, and a cgroup quota if one is set."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        count = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as handle:
            quota, period = handle.read().split()[:2]
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, count)


@dataclass(frozen=True)
class Settings:
    api_key: str
    threads: int
    max_inputs: int
    max_input_chars: int
    max_total_chars: int
    max_tokens: int
    batch_size: int
    batch_tokens: int
    max_pending: int
    dtype: str
    model_path: str
    model_revision: str

    @property
    def max_body_bytes(self) -> int:
        # A JSON escape can take six bytes for one character.
        return self.max_total_chars * 6 + 65536

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)
        api_key = env.get("API_KEY", "")
        if not api_key.strip():
            raise ConfigError(
                "API_KEY is not set. Every request needs it as a bearer token; "
                "set a long random value (for example `openssl rand -hex 32`)."
            )
        if api_key != api_key.strip() or any(c in api_key for c in "\r\n\0"):
            raise ConfigError("API_KEY must not start or end with spaces or hold a line break.")
        dtype = env.get("DTYPE", "bfloat16").strip() or "bfloat16"
        if dtype not in ("bfloat16", "float32"):
            raise ConfigError("DTYPE must be bfloat16 or float32.")
        return Settings(
            api_key=api_key,
            threads=_int(env, "THREADS", available_cpus(), 1, 1024),
            max_inputs=_int(env, "MAX_INPUTS", 64, 1, 2048),
            max_input_chars=_int(env, "MAX_INPUT_CHARS", 32000, 1, 1_000_000),
            max_total_chars=_int(env, "MAX_TOTAL_CHARS", 200000, 1, 10_000_000),
            max_tokens=_int(env, "MAX_TOKENS", 8192, 16, 32768),
            batch_size=_int(env, "BATCH_SIZE", 8, 1, 256),
            batch_tokens=_int(env, "BATCH_TOKENS", 16384, 64, 262144),
            max_pending=_int(env, "MAX_PENDING", 8, 1, 1000),
            dtype=dtype,
            model_path=env.get("MODEL_PATH", "/opt/model"),
            model_revision=env.get("MODEL_REVISION", "unknown"),
        )
