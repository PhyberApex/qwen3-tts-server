# The base image is held on torch 2.6.0 (see renovate.json): the flash-attn
# wheel below is built for exactly this torch, CUDA and C++ ABI, and a newer
# torch has no matching release wheel yet.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime@sha256:f894dae26e1ee8557c544f9cfdb9dc011b1552bf3c1e656b422f2e221d380e40

ARG FLASH_ATTN_VERSION=2.8.3.post2

RUN apt-get update && apt-get install -y --no-install-recommends curl sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# flash-attn on PyPI is a source distribution that compiles for the better part of
# an hour and needs nvcc; the project's GitHub release carries a prebuilt wheel
# for this image's torch 2.6 / CUDA 12 / cxx11 ABI / CPython 3.11, so that is
# what gets installed. Synthesis falls back to default attention if it fails to
# import at runtime (see app.py), so a GPU too old for flash-attn still works.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

COPY app.py .

EXPOSE 8000

# Loading the first model takes well over a minute on a cold cache, so the
# start period is generous; "loading" from /health is a 200 either way.
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=30 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
