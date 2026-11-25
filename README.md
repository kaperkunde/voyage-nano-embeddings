# voyage-nano-embeddings

Self-hosted text embeddings with [MongoDB/voyage-4-nano](https://huggingface.co/MongoDB/voyage-4-nano)
on CPU, behind an OpenAI-compatible `POST /v1/embeddings`. One small
container, one model baked in, no GPU, no network at run time. Made to be
PCP's embeddings provider for search by meaning (see "With PCP"), but it is a
plain OpenAI-style server any client can use.

- **Model.** voyage-4-nano: multilingual, 32k-token context, 2048 native
  dimensions with Matryoshka cuts to 1024, 512 and 256. A query and a document
  are embedded with different prompts (`input_type`).
- **Footprint.** About 1 GB of memory once loaded (1.4 GB peak in the
  measurements below), bf16 weights, an image of about 1.8 GB.
- **Safe by default.** Refuses to start without `API_KEY`, bearer auth with a
  constant-time compare, bounded inputs, runs as uid 10001, no telemetry.

## API

All of it needs `Authorization: Bearer $API_KEY`, except `GET /health` and
`GET /openapi.json`.

### `POST /v1/embeddings`

```sh
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"input": ["Mars is the Red Planet."], "input_type": "document"}'
```

| Field             | Meaning                                                                                                                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `input`           | A string or an array of strings. Required. Not empty, no token arrays.                                                             |
| `input_type`      | Extension. `"query"` for what is searched with, `"document"` for what is searched; `null` or left out is `"document"`.            |
| `dimensions`      | 256, 512, 1024 or 2048. Default **1024**. The vector is cut to this size and made length 1 again.                                  |
| `encoding_format` | `"float"` (default) or `"base64"` (little-endian float32, as OpenAI's).                                                            |
| `model`           | Optional. `voyage-4-nano` or `MongoDB/voyage-4-nano`; anything else is a 400, so a mis-set client finds out at once.               |

The answer is OpenAI's: `{"object": "list", "data": [{"object": "embedding",
"index": 0, "embedding": [...]}], "model": "MongoDB/voyage-4-nano", "usage":
{"prompt_tokens": n, "total_tokens": n}}`. `usage` counts the tokens the model
read, the model's short prompt included. Vectors have length 1, so the dot
product is the cosine.

Search needs both kinds: embed the documents with `input_type: "document"`
once, and each search with `"query"`. Mixing them up still works, with worse
results. Use one `dimensions` for both.

Errors are OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:
400 for a bad request (before the model runs), 401 for the key (with
`WWW-Authenticate: Bearer`), 413 for a body past the limit, 429 when too many
requests are waiting (with `Retry-After`).

### `GET /health`

No key. `{"status": "ok", "model": "MongoDB/voyage-4-nano", "revision": "<commit>",
"dim": 1024, "dimensions": [256, 512, 1024, 2048]}`. The server listens only
after the model has loaded and run once, so `ok` means it can answer.

### `GET /openapi.json`

No key. The OpenAPI 3.0 description of `/v1/embeddings`, with its bearer
security scheme. PCP registers the service's API from it.

## Settings

Environment variables, all read once at start.

| Variable          | Default           | Meaning                                                                                     |
| ----------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| `API_KEY`         | none, **required** | The bearer token. The server exits at start without it.                                    |
| `THREADS`         | all cores         | Threads the model uses (affinity and a container's CPU quota counted).                      |
| `MAX_INPUTS`      | 64                | Texts per request.                                                                          |
| `MAX_INPUT_CHARS` | 32000             | Characters in one text.                                                                     |
| `MAX_TOTAL_CHARS` | 200000            | Characters in all of a request's texts. The body may be six times that.                     |
| `MAX_TOKENS`      | 8192              | A longer text is cut to this many tokens (the model takes up to 32768; memory grows with it). |
| `BATCH_SIZE`      | 8                 | Texts the model reads at once. Texts are sorted by length first, so batches pad little.     |
| `BATCH_TOKENS`    | 16384             | Padded tokens in one batch, at most.                                                        |
| `MAX_PENDING`     | 8                 | Requests running or waiting; the next is a 429. The model runs one request at a time.       |
| `DTYPE`           | `bfloat16`        | `bfloat16` (half the memory; fast on CPUs with AVX-512 BF16 or AMX) or `float32`.           |

## Running it

```sh
docker run -d --name voyage-nano -p 127.0.0.1:8000:8000 \
  -e API_KEY="$(openssl rand -hex 32)" \
  --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp \
  ghcr.io/kaperkunde/voyage-nano-embeddings@sha256:<digest>
```

Podman takes the same flags. The image is signed with cosign (keyless) and its
build is attested; pull by digest, the one in the workflow run's summary:

```sh
cosign verify ghcr.io/kaperkunde/voyage-nano-embeddings@sha256:<digest> \
  --certificate-identity-regexp '^https://github\.com/kaperkunde/voyage-nano-embeddings/\.github/workflows/image\.yml@refs/(heads/main|tags/v[0-9]+\.[0-9]+\.[0-9]+)$' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

It starts in a few seconds; `/health` answers once the model is loaded. The
container needs a writable `/tmp` (the model's code is staged there), which the
flags above give it. Only `linux/amd64` is built: the locked, hash-checked
requirements are for that platform.

## With PCP

It is a PCP common service, [`voyage-4-nano`](https://github.com/kaperkunde/pcp-gg-common):
launch it on an SSH server of yours and PCP

- makes the `API_KEY` itself, a long random one, and sends it for you;
- adds the service's API as an endpoint from `/openapi.json`;
- can use that endpoint as the embeddings provider for search (Settings →
  Search). PCP's Voyage AI style sends `input_type` (`query` or `document`);
  for an OpenAI-style provider it sends it only where that is switched on.
  Without it every text is embedded as a document: search still works, a little
  less well for queries.

The item in this repository, [`pcp-gg-common/services/voyage-4-nano.yaml`](pcp-gg-common/services/voyage-4-nano.yaml),
is what goes into that repository once the image's digest is known.

## Resources, measured

Measured on a shared 4-vCPU Intel Xeon (Sapphire Rapids: AVX-512 BF16 and AMX)
with 16 GB of RAM, bf16, `BATCH_SIZE=4`, while other work kept its load average
near 35, so these are slow and noisy: take them as an upper bound. Median of
repeated calls after a warm-up, over loopback (`scripts/bench.py`).

| Request                                  | Time   |
| ---------------------------------------- | ------ |
| 1 query, about 12 tokens                 | 0.15 s |
| 1 document, about 200 tokens             | 0.40 s |
| 50 short sentences (one request)         | 3.0 s  |
| 20 documents of about 200 tokens         | 8.2 s  |
| 1 document of about 2000 tokens          | 7.7 s  |

Memory: about 1.0 GB resident after loading, 1.4 GB at the peak of these runs.
The weights are 672 MB in bf16; `float32` would use about 2.7 GB for them
alone, which this machine could not spare for a test.

## Security notes

- **The model's code runs.** voyage-4-nano needs `trust_remote_code`: loading
  it executes `modeling_qwen3_bidirectional.py` from the model's repository.
  That is why the model's revision is pinned to a commit (the `MODEL_REVISION`
  build argument in the `Dockerfile`) and baked in at build; nothing is fetched
  at run time (`HF_HUB_OFFLINE=1`). A newer revision is a change to read first,
  that file above all.
- **The build is locked.** The base image is pinned by digest; Python packages
  by version and hash (`requirements.txt`, from `requirements.in`); CPU-only
  PyTorch comes from PyTorch's own CPU index; GitHub Actions are pinned by
  commit.
- **No outbound traffic.** Telemetry is off, and the server makes no request.
  Run it with no network if you like; the health check is local.
- **Auth.** The key is compared as SHA-256 digests with `hmac.compare_digest`.
  It is the only credential: put the service behind TLS if it leaves a
  trusted network. Request bodies, texts and vectors are never logged; an
  unexpected failure logs the error's type and where it happened, not its text.
- **Bounded.** Body, inputs, characters, tokens and waiting requests are all
  limited (see Settings); a text is validated before the model sees it.
- **One request at a time.** The model uses all the threads for one request;
  others wait, up to `MAX_PENDING`, then get a 429. It is a service for one
  owner's tools, not a public endpoint.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt     # no torch, no model
ruff check . && ruff format --check . && pytest
```

The tests use a small fake encoder, so they download nothing. To run the real
thing, install `requirements.txt` (see its header for the command), download
the model at the pinned revision into a folder, and start it:

```sh
API_KEY=... MODEL_PATH=/path/to/model uvicorn --factory app.server:build
API_KEY=... python scripts/smoke.py http://127.0.0.1:8000   # also: scripts/bench.py
```

Updating `requirements.txt`: edit `requirements.in` and run the `uv pip compile`
command at the top of that file.

## License

The code is Apache-2.0 (see `LICENSE`). The model, MongoDB/voyage-4-nano
(Voyage AI), is Apache-2.0 too, per its model card and `LICENSE.txt`; see `NOTICE`.
