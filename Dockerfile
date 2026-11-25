# syntax=docker/dockerfile:1
#
# voyage-nano-embeddings: MongoDB/voyage-4-nano on CPU behind an
# OpenAI-compatible /v1/embeddings. The model is baked in at build (no
# network at run time); runs as uid 10001, never root.
#
#   docker run -d --name voyage-nano -p 127.0.0.1:8000:8000 \
#     -e API_KEY --cap-drop ALL --security-opt no-new-privileges \
#     ghcr.io/kaperkunde/voyage-nano-embeddings@sha256:<digest>
#
# See README.md.

# python:3.12-slim, by digest (Debian, amd64 and arm64 in one index).
ARG PYTHON_IMAGE=docker.io/library/python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
# The model's commit on Hugging Face. The model needs trust_remote_code: it
# runs the Python file modeling_qwen3_bidirectional.py from this revision, so
# the revision is pinned and a new one is a change here, read first.
ARG MODEL_REVISION=587d8d6dcaec029a9779c2098d6af020e4a07c01

# ---- deps: the Python packages, exactly as locked, hashes checked ------------
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY requirements.txt ./
# --require-hashes: every file is checked against requirements.txt, and no
# package outside it is taken (--no-deps). torch comes from PyTorch's CPU
# index: the CUDA libraries would add gigabytes this service never uses.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --no-deps \
       --index-url https://download.pytorch.org/whl/cpu \
       --extra-index-url https://pypi.org/simple \
       -r requirements.txt \
    && find /opt/venv -name '__pycache__' -prune -exec rm -rf {} + \
    && rm -rf /opt/venv/lib/python3.12/site-packages/torch/include \
              /opt/venv/lib/python3.12/site-packages/torch/test \
              /opt/venv/lib/python3.12/site-packages/pip

# ---- model: the pinned revision, downloaded once -----------------------------
FROM deps AS model
ARG MODEL_REVISION
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN /opt/venv/bin/python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "MongoDB/voyage-4-nano",
    revision="${MODEL_REVISION}",
    local_dir="/opt/model",
    # The weights and what loads them: not the README, the images, the
    # ONNX or other formats a revision may add.
    allow_patterns=["*.json", "*.py", "*.txt", "model.safetensors"],
)
PY
RUN rm -rf /opt/model/.cache && ls -l /opt/model

# ---- run ------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
ARG MODEL_REVISION
RUN groupadd --gid 10001 embeddings \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin embeddings
COPY --from=deps /opt/venv /opt/venv
COPY --from=model /opt/model /opt/model
COPY app /app/app
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/opt/model \
    MODEL_REVISION=${MODEL_REVISION} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HF_HOME=/tmp/hf \
    HF_MODULES_CACHE=/tmp/hf/modules \
    TOKENIZERS_PARALLELISM=false
USER 10001:10001
EXPOSE 8000
# The model loads before the server listens, which takes a few seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
LABEL org.opencontainers.image.title="voyage-nano-embeddings" \
      org.opencontainers.image.description="MongoDB/voyage-4-nano text embeddings on CPU, OpenAI-compatible" \
      org.opencontainers.image.licenses="Apache-2.0"
CMD ["uvicorn", "--factory", "app.server:build", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
