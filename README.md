# qwen3-tts-server

A small HTTP server around [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): one Python
file on FastAPI that exposes an **OpenAI-compatible `POST /v1/audio/speech`**, swaps between
the three 1.7B model variants on demand (preset voices, voice design from a description, voice
cloning from reference audio), caches voice-clone prompts so a cloned voice is computed once,
and reports what it is doing on `/metrics`, `/status` and `/health`.

It exists because the official Qwen3-TTS repo gives you a Python API and a Gradio demo, and
what a homelab wants is a container that speaks the OpenAI audio API so every client that
already talks to OpenAI, or to another OpenAI-shaped TTS server, works unchanged.

## Trust model, read this first

**There is no authentication unless you set `TTS_API_KEY`.** Without it, anyone who can reach
the port can synthesise speech, swap the loaded model and change the log level, and the server
logs one warning at startup saying so. Run it on a private network or behind a reverse proxy
that authenticates, or set the key: then every route except `/health` and `/metrics` requires
`Authorization: Bearer <key>`, which the OpenAI SDKs send anyway.

## Running

The image is published to GHCR for `linux/amd64` with CUDA 12.6 only. It needs an NVIDIA GPU
with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. `TTS_DEVICE=cpu` is accepted and works, at a speed you will not enjoy.

Tags: `:latest` and `:x.y.z` for releases, `:next` for the current `main`.

```yaml
services:
  qwen3-tts:
    image: ghcr.io/phyberapex/qwen3-tts-server:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DEFAULT_TASK_TYPE=Base
      - TTS_API_KEY=change-me
    volumes:
      - hf-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  hf-cache:
```

The `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=0` form works too. The models are downloaded
from Hugging Face on first use into `/root/.cache/huggingface`, so mount a volume there or you
pay the download on every container recreate. Each variant is about 4.3 GB on disk; all three
cached is 13 GB.

### Sizing

Measured on an RTX 3060 12 GB with the 1.7B models in `bfloat16`:

| | |
|---|---|
| VRAM with one model loaded | ~4.8 GB |
| First load from a warm disk cache | 30–60 s |
| Model swap (`/switch` or a request for another `task_type`) | the same, plus unloading |

Only one model is resident at a time; a request for a different `task_type` unloads the
current one first. If your clients mix modes, expect that swap on every change and consider
one container per mode.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TTS_API_KEY` | unset | When set, every route except `/health` and `/metrics` requires `Authorization: Bearer <key>` |
| `DEFAULT_TASK_TYPE` | `Base` | Model loaded at startup: `CustomVoice`, `VoiceDesign` or `Base` |
| `TTS_DEVICE` | `cuda:0` | Passed to `device_map`; `cuda:1`, `cpu`, … |
| `TTS_DTYPE` | `bfloat16` | `bfloat16`, `float16` or `float32`. Pre-Ampere cards (sm < 80) want `float16` |
| `TTS_MODEL_CUSTOM_VOICE` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Hugging Face id or local path for the preset-voice model |
| `TTS_MODEL_VOICE_DESIGN` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | … for the voice-design model |
| `TTS_MODEL_BASE` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | … for the voice-clone model |
| `MAX_CACHED_PROMPTS` | `16` | Voice-clone prompts kept in memory; least recently used is evicted |
| `HF_TOKEN` | unset | Only needed if you point a model variable at a gated or private repo |

Flash Attention 2 is installed and used when the GPU supports it (Ampere and newer). On an
older card the load logs `flash_attention_2 unavailable` once and falls back to the default
attention; nothing else changes.

## API

### `POST /v1/audio/speech`

Returns `audio/wav` (16-bit PCM). The body is the OpenAI shape plus a few additive fields:

| Field | Default | Meaning |
|---|---|---|
| `input` | required | The text to speak |
| `task_type` | `CustomVoice` | Which model answers: `CustomVoice`, `VoiceDesign` or `Base` |
| `language` | `Auto` | Language hint passed to the model |
| `voice` | `Vivian` | `CustomVoice` only: one of the preset speakers from `/v1/audio/voices` |
| `instructions` | | `CustomVoice`: a style instruction. `VoiceDesign`: the description of the voice to invent |
| `ref_audio` | | `Base` only: base64 (optionally a `data:` URL) of the reference recording to clone |
| `ref_text` | | `Base` only: the transcript of `ref_audio`, which improves the clone |
| `x_vector_only_mode` | | `Base` only: clone timbre only, from the speaker embedding, no transcript needed |
| `prompt_name` | | `Base` only: use a prompt cached via `POST /prompts` instead of sending `ref_audio` |
| `model`, `response_format` | | Accepted for OpenAI compatibility and ignored; output is always WAV |

```bash
# Preset voice
curl -s localhost:8000/v1/audio/speech \
  -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d '{"input":"Good morning from the workshop.","task_type":"CustomVoice","voice":"Ryan"}' \
  -o hello.wav

# A voice designed from a description
curl -s localhost:8000/v1/audio/speech \
  -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d '{"input":"All systems nominal.","task_type":"VoiceDesign","instructions":"calm, low, slightly amused, a little gravel"}' \
  -o designed.wav

# Cloned from a reference recording
curl -s localhost:8000/v1/audio/speech \
  -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d "{\"input\":\"This is my cloned voice.\",\"task_type\":\"Base\",\"ref_audio\":\"$(base64 -w0 me.wav)\",\"ref_text\":\"what I said in me.wav\"}" \
  -o cloned.wav
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="change-me")
audio = client.audio.speech.create(
    model="qwen3-tts",
    voice="Serena",
    input="Hello from a local model.",
    extra_body={"task_type": "CustomVoice", "language": "English"},
)
audio.write_to_file("hello.wav")
```

Requests are synthesised one at a time under a lock. A second request waits; it does not
fail. `qwen3_tts_requests_waiting` on `/metrics` tells you how long the line is.

### `GET /v1/audio/voices`

The preset speakers the `CustomVoice` model ships: Aiden, Dylan, Eric, Ono_Anna, Ryan, Serena,
Sohee, Uncle_fu, Vivian.

### Voice-clone prompts

Cloning from `ref_audio` recomputes the speaker prompt on every request. For a voice you reuse,
create it once and refer to it by name:

```bash
curl -s localhost:8000/prompts -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d "{\"name\":\"me\",\"ref_audio\":\"$(base64 -w0 me.wav)\",\"ref_text\":\"what I said in me.wav\"}"

curl -s localhost:8000/v1/audio/speech -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d '{"input":"Same voice, no upload.","task_type":"Base","prompt_name":"me"}' -o again.wav

curl -s localhost:8000/prompts -H 'Authorization: Bearer change-me'
# {"prompts":["me"],"count":1,"max_cached":16}
```

Prompts live in process memory: a restart forgets them, and the cache holds
`MAX_CACHED_PROMPTS` with least-recently-used eviction. Creating a prompt loads the `Base`
model if another one is resident.

### Management

| Route | What it does |
|---|---|
| `POST /switch` `{"task_type": "VoiceDesign"}` | Load that model now instead of on the next request; returns `switch_time_ms` |
| `POST /log-level` `{"level": "debug"}` | `debug`, `info`, `warning` or `error`, at runtime |
| `GET /status` | `idle` or `synthesizing`, the running and waiting counts, the loaded model and the log level |
| `GET /health` | `{"status": "ok" \| "loading", "model_loaded": …, "ready": bool}`. Always open. 200 while loading, so point a readiness probe at `ready` |
| `GET /metrics` | Prometheus exposition. Always open |

### Metrics

```text
# TYPE qwen3_tts_requests_running gauge
qwen3_tts_requests_running 1
# TYPE qwen3_tts_requests_waiting gauge
qwen3_tts_requests_waiting 2
# TYPE qwen3_tts_model_loaded gauge
qwen3_tts_model_loaded{task_type="CustomVoice"} 0
qwen3_tts_model_loaded{task_type="VoiceDesign"} 0
qwen3_tts_model_loaded{task_type="Base"} 1
```

`qwen3_tts_model_loaded` is a one-hot across the three task types, so `sum(...)` is 1 when a
model is resident and 0 during a swap or before the first load.

## Running outside Docker

Python 3.11, a CUDA-capable torch, then:

```bash
pip install -r requirements.txt
# optional: pip install flash-attn, matching your torch build
uvicorn app:app --host 0.0.0.0 --port 8000
```

`sox` is needed on the host for some reference-audio formats (`apt install sox libsox-fmt-all`).

## Developing

```bash
pip install fastapi httpx
python -m unittest discover -p 'test_*.py' -v
```

The tests stub `torch`, `numpy` and `soundfile` in-process, so they run anywhere with no GPU
and none of the model dependencies. CI runs them on every pull request and then builds the
image and imports the app and flash-attn inside it.

The Dockerfile holds `pytorch/pytorch` on 2.6.0 on purpose: flash-attn's GitHub release
carries a prebuilt wheel for that torch, CUDA 12 and C++ ABI, and installing it is what keeps
the image build at minutes instead of the hour-long compile the PyPI sdist needs. Renovate
moves the base image's digest and the flash-attn version; the torch tag moves by hand once a
flash-attn release ships a wheel for a newer torch.

## License

[MIT](LICENSE). Qwen3-TTS itself is Apache-2.0, see the model cards on Hugging Face.
