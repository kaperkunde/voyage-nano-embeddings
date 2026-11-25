"""Entry point: `uvicorn --factory app.server:build`. Loads the real model."""

from __future__ import annotations

import os

from .api import create_app
from .config import Settings


def build():
    settings = Settings.from_env()
    # Set before torch loads, so its thread pools start at this size.
    os.environ["OMP_NUM_THREADS"] = str(settings.threads)
    os.environ["MKL_NUM_THREADS"] = str(settings.threads)
    from .encoder import VoyageEncoder

    encoder = VoyageEncoder(settings)
    # One short run, so the first request does not pay for the first
    # kernels, and the service is healthy only once the model answers.
    encoder.encode(["warm up"], "query")
    return create_app(encoder, settings)
