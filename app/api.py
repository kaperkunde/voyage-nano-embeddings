"""The HTTP API: OpenAI's /v1/embeddings, /health and /openapi.json."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .config import (
    DEFAULT_DIMENSIONS,
    DIMENSIONS,
    MODEL_ID,
    MODEL_NAMES,
    Settings,
)
from .encoder import Encoder

log = logging.getLogger("voyage-nano")


class ApiError(Exception):
    """An answer in OpenAI's error shape."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str = "invalid_request_error",
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.param = param
        self.headers = headers or {}


def error_response(error: ApiError) -> JSONResponse:
    kind = "invalid_request_error" if error.status < 500 else "server_error"
    if error.status == 401:
        kind = "authentication_error"
    if error.status == 429:
        kind = "rate_limit_error"
    return JSONResponse(
        {
            "error": {
                "message": error.message,
                "type": kind,
                "param": error.param,
                "code": error.code,
            }
        },
        status_code=error.status,
        headers=error.headers,
    )


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: list[str]
    input_type: str
    dimensions: int
    base64: bool


def parse_request(raw: bytes, settings: Settings) -> EmbeddingRequest:
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError):
        raise ApiError(400, "The body is not valid JSON.", code="invalid_json") from None
    if not isinstance(body, dict):
        raise ApiError(400, "The body must be a JSON object.", code="invalid_body")

    if "input" not in body:
        raise ApiError(400, "`input` is required.", param="input", code="missing_input")
    value = body["input"]
    texts = [value] if isinstance(value, str) else value
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ApiError(
            400,
            "`input` must be a string or an array of strings (token arrays are not supported).",
            param="input",
            code="invalid_input",
        )
    if not texts:
        raise ApiError(400, "`input` must not be empty.", param="input", code="invalid_input")
    if len(texts) > settings.max_inputs:
        raise ApiError(
            400,
            f"At most {settings.max_inputs} inputs per request, got {len(texts)}.",
            param="input",
            code="too_many_inputs",
        )
    total = 0
    for index, text in enumerate(texts):
        if not text.strip():
            raise ApiError(400, f"input[{index}] is empty.", param="input", code="empty_input")
        if len(text) > settings.max_input_chars:
            raise ApiError(
                400,
                f"input[{index}] is {len(text)} characters; the limit is "
                f"{settings.max_input_chars} per input.",
                param="input",
                code="input_too_long",
            )
        total += len(text)
    if total > settings.max_total_chars:
        raise ApiError(
            400,
            f"The inputs add up to {total} characters; the limit is "
            f"{settings.max_total_chars} per request.",
            param="input",
            code="request_too_long",
        )

    model = body.get("model")
    if model is not None and model not in MODEL_NAMES:
        raise ApiError(
            400,
            f"This server serves {MODEL_ID} only (as {', '.join(MODEL_NAMES)}); "
            "leave `model` out or use one of those.",
            param="model",
            code="model_not_found",
        )

    fmt = body.get("encoding_format")
    if fmt not in (None, "float", "base64"):
        raise ApiError(
            400,
            '`encoding_format` must be "float" or "base64".',
            param="encoding_format",
            code="invalid_encoding_format",
        )

    dimensions = body.get("dimensions")
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS
    elif isinstance(dimensions, bool) or dimensions not in DIMENSIONS:
        raise ApiError(
            400,
            f"`dimensions` must be one of {', '.join(map(str, DIMENSIONS))}.",
            param="dimensions",
            code="invalid_dimensions",
        )

    input_type = body.get("input_type")
    if input_type not in (None, "query", "document"):
        raise ApiError(
            400,
            '`input_type` must be "query", "document" or null.',
            param="input_type",
            code="invalid_input_type",
        )

    return EmbeddingRequest(
        texts=texts,
        input_type=input_type or "document",
        dimensions=int(dimensions),
        base64=fmt == "base64",
    )


def truncate(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    """The first `dimensions` of each row (a Matryoshka cut), back to length 1."""
    cut = np.ascontiguousarray(vectors[:, :dimensions], dtype=np.float32)
    norms = np.linalg.norm(cut, axis=1, keepdims=True)
    return cut / np.maximum(norms, 1e-12)


def openapi_document() -> dict[str, Any]:
    error = {"$ref": "#/components/schemas/Error"}

    def failure(text: str) -> dict[str, Any]:
        return {"description": text, "content": {"application/json": {"schema": error}}}

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "voyage-nano-embeddings",
            "version": "1.0.0",
            "description": "OpenAI-compatible text embeddings with MongoDB/voyage-4-nano on CPU.",
        },
        "security": [{"bearerAuth": []}],
        "paths": {
            "/v1/embeddings": {
                "post": {
                    "operationId": "createEmbedding",
                    "summary": "Create embeddings",
                    "description": (
                        "Turns each input text into a unit vector. Set `input_type` to "
                        "`query` for a search query and `document` (the default) for "
                        "the text searched; the model embeds each differently."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/EmbeddingRequest"}
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "One embedding per input, in order.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/EmbeddingResponse"}
                                }
                            },
                        },
                        "400": failure("The request is not valid."),
                        "401": failure("The bearer token is missing or wrong."),
                        "413": failure("The body is too large."),
                        "429": failure("Too many requests at once; retry after a moment."),
                    },
                }
            }
        },
        "components": {
            "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
            "schemas": {
                "EmbeddingRequest": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {
                            "description": "The text to embed: one string or an array of strings.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional; voyage-4-nano or MongoDB/voyage-4-nano.",
                        },
                        "input_type": {
                            "type": "string",
                            "nullable": True,
                            "enum": ["query", "document"],
                            "description": (
                                "query or document; left out, the text is embedded as a document."
                            ),
                        },
                        "dimensions": {
                            "type": "integer",
                            "enum": list(DIMENSIONS),
                            "default": DEFAULT_DIMENSIONS,
                            "description": "Length of each vector (a Matryoshka size).",
                        },
                        "encoding_format": {
                            "type": "string",
                            "enum": ["float", "base64"],
                            "default": "float",
                            "description": "float lists, or base64 of little-endian float32.",
                        },
                    },
                },
                "EmbeddingResponse": {
                    "type": "object",
                    "required": ["object", "data", "model", "usage"],
                    "properties": {
                        "object": {"type": "string", "enum": ["list"]},
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["object", "index", "embedding"],
                                "properties": {
                                    "object": {"type": "string", "enum": ["embedding"]},
                                    "index": {"type": "integer"},
                                    "embedding": {
                                        "description": "Numbers, or a base64 string.",
                                        "oneOf": [
                                            {"type": "array", "items": {"type": "number"}},
                                            {"type": "string"},
                                        ],
                                    },
                                },
                            },
                        },
                        "model": {"type": "string"},
                        "usage": {
                            "type": "object",
                            "properties": {
                                "prompt_tokens": {"type": "integer"},
                                "total_tokens": {"type": "integer"},
                            },
                        },
                    },
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "type": {"type": "string"},
                                "param": {"type": "string", "nullable": True},
                                "code": {"type": "string"},
                            },
                        }
                    },
                },
            },
        },
    }


def create_app(encoder: Encoder, settings: Settings) -> FastAPI:
    app = FastAPI(title="voyage-nano-embeddings", docs_url=None, redoc_url=None, openapi_url=None)
    key_digest = hashlib.sha256(settings.api_key.encode()).digest()
    lock = asyncio.Lock()  # the model runs one request at a time
    app.state.pending = 0  # requests running or waiting for the model
    document = openapi_document()

    def authorize(request: Request) -> None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        given = hashlib.sha256(token.strip().encode()).digest()
        # Compared as digests, so the comparison is constant-time whatever
        # the lengths are.
        ok = hmac.compare_digest(given, key_digest)
        if scheme.lower() != "bearer" or not ok:
            raise ApiError(
                401,
                "Missing or wrong bearer token.",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(ApiError)
    async def on_api_error(_: Request, error: ApiError) -> JSONResponse:
        return error_response(error)

    @app.exception_handler(404)
    async def on_missing(_: Request, __: Exception) -> JSONResponse:
        return error_response(ApiError(404, "Not found.", code="not_found"))

    @app.exception_handler(405)
    async def on_method(_: Request, __: Exception) -> JSONResponse:
        return error_response(ApiError(405, "Method not allowed.", code="method_not_allowed"))

    @app.exception_handler(Exception)
    async def on_failure(_: Request, error: Exception) -> JSONResponse:
        # The type and the frames, never the message: it could hold the text.
        frames = "".join(traceback.format_tb(error.__traceback__))
        log.error("request failed: %s\n%s", type(error).__name__, frames)
        return error_response(ApiError(500, "The server failed.", code="server_error"))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "revision": settings.model_revision,
            "dim": DEFAULT_DIMENSIONS,
            "dimensions": list(DIMENSIONS),
        }

    @app.get("/openapi.json")
    async def openapi() -> dict[str, Any]:
        return document

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        authorize(request)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_body_bytes:
            raise ApiError(413, "The body is too large.", code="body_too_large")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_body_bytes:
                raise ApiError(413, "The body is too large.", code="body_too_large")
            chunks.append(chunk)
        parsed = parse_request(b"".join(chunks), settings)

        if app.state.pending >= settings.max_pending:
            raise ApiError(
                429,
                "Too many requests waiting; retry in a few seconds.",
                code="rate_limit_exceeded",
                headers={"Retry-After": "5"},
            )
        app.state.pending += 1
        try:
            async with lock:
                vectors, tokens = await run_in_threadpool(
                    encoder.encode, parsed.texts, parsed.input_type
                )
        finally:
            app.state.pending -= 1

        vectors = truncate(vectors, parsed.dimensions)
        data = []
        for index, vector in enumerate(vectors):
            embedding: Any
            if parsed.base64:
                embedding = base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
            else:
                embedding = vector.tolist()
            data.append({"object": "embedding", "index": index, "embedding": embedding})
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": MODEL_ID,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    return app
