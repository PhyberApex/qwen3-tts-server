"""
Qwen3-TTS server.

A small FastAPI wrapper around the qwen-tts library that exposes an
OpenAI-compatible /v1/audio/speech endpoint, hot-swaps between the three
Qwen3-TTS model variants and caches voice-clone prompts.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("qwen3-tts")

LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_current_log_level = "info"

DEFAULT_TASK_MODELS: dict[str, str] = {
    "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

TASK_MODEL_ENV_VARS: dict[str, str] = {
    "CustomVoice": "TTS_MODEL_CUSTOM_VOICE",
    "VoiceDesign": "TTS_MODEL_VOICE_DESIGN",
    "Base": "TTS_MODEL_BASE",
}

SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")


def task_models_from_env(environ: dict[str, str]) -> dict[str, str]:
    return {
        task_type: environ.get(TASK_MODEL_ENV_VARS[task_type]) or default
        for task_type, default in DEFAULT_TASK_MODELS.items()
    }


def resolve_dtype(name: str):
    if name not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported TTS_DTYPE: {name!r} (one of {', '.join(SUPPORTED_DTYPES)})"
        )
    return getattr(torch, name)


TASK_MODELS: dict[str, str] = task_models_from_env(os.environ)
DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
DTYPE_NAME = os.environ.get("TTS_DTYPE", "bfloat16")
API_KEY = os.environ.get("TTS_API_KEY") or None
OPEN_PATHS = frozenset({"/health", "/metrics"})

CUSTOM_VOICES = [
    {"name": "Aiden"},
    {"name": "Dylan"},
    {"name": "Eric"},
    {"name": "Ono_Anna"},
    {"name": "Ryan"},
    {"name": "Serena"},
    {"name": "Sohee"},
    {"name": "Uncle_fu"},
    {"name": "Vivian"},
]

DEFAULT_TASK = os.environ.get("DEFAULT_TASK_TYPE", "Base")
MAX_CACHED_PROMPTS = int(os.environ.get("MAX_CACHED_PROMPTS", "16"))


class TTSEngine:
    """Manages Qwen3-TTS model lifecycle, synthesis, and prompt caching."""

    def __init__(self) -> None:
        self.model = None
        self.current_task_type: str | None = None
        self._lock = threading.Lock()
        self._waiting_lock = threading.Lock()
        self.requests_running = 0
        self.requests_waiting = 0
        self.prompts: OrderedDict = OrderedDict()
        self.ready = False

    def load_model(self, task_type: str) -> None:
        if task_type not in TASK_MODELS:
            raise ValueError(f"Unknown task_type: {task_type}")

        if self.model is not None:
            logger.info("Unloading model: %s", self.current_task_type)
            del self.model
            self.model = None
            if DEVICE.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

        model_name = TASK_MODELS[task_type]
        logger.info("Loading model: %s (%s) on %s as %s", task_type, model_name, DEVICE, DTYPE_NAME)
        start = time.monotonic()

        from qwen_tts import Qwen3TTSModel

        dtype = resolve_dtype(DTYPE_NAME)
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=DEVICE,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            logger.warning("flash_attention_2 unavailable, using default attention")
            self.model = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map=DEVICE,
                dtype=dtype,
            )

        self.current_task_type = task_type
        logger.info("Model loaded in %.1fs: %s", time.monotonic() - start, task_type)

    def _ensure_model(self, task_type: str) -> None:
        if self.current_task_type != task_type:
            self.load_model(task_type)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        task_type: str,
        text: str,
        language: str = "Auto",
        voice: str | None = None,
        instructions: str | None = None,
        ref_audio_bytes: bytes | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool | None = None,
        prompt_name: str | None = None,
    ) -> bytes:
        with self._waiting_lock:
            self.requests_waiting += 1
        with self._lock:
            with self._waiting_lock:
                self.requests_waiting -= 1
            self.requests_running += 1
            try:
                self._ensure_model(task_type)
                wavs, sr = self._generate(
                    task_type,
                    text,
                    language,
                    voice,
                    instructions,
                    ref_audio_bytes,
                    ref_text,
                    x_vector_only_mode,
                    prompt_name,
                )
                return self._encode_wav(wavs[0], sr)
            finally:
                self.requests_running -= 1

    def _generate(
        self,
        task_type: str,
        text: str,
        language: str,
        voice: str | None,
        instructions: str | None,
        ref_audio_bytes: bytes | None,
        ref_text: str | None,
        x_vector_only_mode: bool | None,
        prompt_name: str | None,
    ):
        if task_type == "CustomVoice":
            kwargs: dict = {
                "text": text,
                "language": language,
                "speaker": voice or "Vivian",
            }
            if instructions:
                kwargs["instruct"] = instructions
            return self.model.generate_custom_voice(**kwargs)

        if task_type == "VoiceDesign":
            return self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=instructions or "",
            )

        if task_type == "Base":
            kwargs = {"text": text, "language": language}

            if prompt_name and prompt_name in self.prompts:
                self.prompts.move_to_end(prompt_name)
                kwargs["voice_clone_prompt"] = self.prompts[prompt_name]
            elif ref_audio_bytes:
                kwargs["ref_audio"] = self._decode_audio(ref_audio_bytes)
                if ref_text:
                    kwargs["ref_text"] = ref_text
                if x_vector_only_mode is not None:
                    kwargs["x_vector_only_mode"] = x_vector_only_mode
            elif prompt_name:
                raise ValueError(f"Cached prompt not found: {prompt_name}")
            else:
                raise ValueError("Base mode requires ref_audio or prompt_name")

            return self.model.generate_voice_clone(**kwargs)

        raise ValueError(f"Unknown task_type: {task_type}")

    # ------------------------------------------------------------------
    # Prompt caching (for reusable voice clone sources)
    # ------------------------------------------------------------------

    def create_prompt(
        self,
        name: str,
        ref_audio_bytes: bytes,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
    ) -> None:
        with self._lock:
            self._ensure_model("Base")
            kwargs: dict = {
                "ref_audio": self._decode_audio(ref_audio_bytes),
                "x_vector_only_mode": x_vector_only_mode,
            }
            if ref_text:
                kwargs["ref_text"] = ref_text
            self.prompts[name] = self.model.create_voice_clone_prompt(**kwargs)
            self.prompts.move_to_end(name)
            if len(self.prompts) > MAX_CACHED_PROMPTS:
                evicted, _ = self.prompts.popitem(last=False)
                logger.info(
                    "Evicted LRU voice clone prompt: %s (cap=%d)",
                    evicted,
                    MAX_CACHED_PROMPTS,
                )
            logger.info(
                "Cached voice clone prompt: %s (%d/%d)",
                name,
                len(self.prompts),
                MAX_CACHED_PROMPTS,
            )

    def list_prompts(self) -> list[str]:
        with self._lock:
            return list(self.prompts.keys())

    # ------------------------------------------------------------------
    # Model switching (explicit pre-load without synthesis)
    # ------------------------------------------------------------------

    def switch_model(self, task_type: str) -> float:
        with self._lock:
            start = time.monotonic()
            self._ensure_model(task_type)
            return time.monotonic() - start

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
        return sf.read(io.BytesIO(audio_bytes))

    @staticmethod
    def _encode_wav(audio_array: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


engine = TTSEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if API_KEY is None:
        logger.warning(
            "TTS_API_KEY is not set: every route, including /switch and /log-level, "
            "is open to anyone who can reach this port"
        )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load_model, DEFAULT_TASK)
    engine.ready = True
    logger.info("Service ready (default model: %s)", DEFAULT_TASK)
    yield


app = FastAPI(title="Qwen3-TTS", lifespan=lifespan)


def is_authorized(authorization: str | None) -> bool:
    if API_KEY is None:
        return True
    if not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    return scheme.lower() == "bearer" and token.strip() == API_KEY


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if request.url.path in OPEN_PATHS or is_authorized(request.headers.get("authorization")):
        return await call_next(request)
    return JSONResponse(
        status_code=401,
        content={"detail": "Missing or invalid bearer token"},
        headers={"WWW-Authenticate": "Bearer"},
    )


# ======================================================================
# OpenAI-compatible endpoints
# ======================================================================


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    response_format: str = "wav"
    task_type: str = "CustomVoice"
    language: str = "Auto"
    voice: str | None = None
    instructions: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool | None = None
    prompt_name: str | None = None


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.task_type not in TASK_MODELS:
        raise HTTPException(400, detail=f"Unknown task_type: {req.task_type}")

    ref_audio_bytes = None
    if req.ref_audio:
        raw = req.ref_audio
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1]
        ref_audio_bytes = base64.b64decode(raw)

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            None,
            engine.synthesize,
            req.task_type,
            req.input,
            req.language,
            req.voice,
            req.instructions,
            ref_audio_bytes,
            req.ref_text,
            req.x_vector_only_mode,
            req.prompt_name,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Synthesis failed")
        raise HTTPException(500, detail=str(e)) from e

    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/v1/audio/voices")
async def voices():
    return {"voices": CUSTOM_VOICES}


# ======================================================================
# Management endpoints
# ======================================================================


class SwitchRequest(BaseModel):
    task_type: str


@app.post("/switch")
async def switch(req: SwitchRequest):
    if req.task_type not in TASK_MODELS:
        raise HTTPException(400, detail=f"Unknown task_type: {req.task_type}")

    loop = asyncio.get_event_loop()
    try:
        elapsed = await loop.run_in_executor(
            None, engine.switch_model, req.task_type
        )
    except Exception as e:
        logger.exception("Model switch failed")
        raise HTTPException(500, detail=str(e)) from e

    return {
        "status": "ok",
        "model_loaded": req.task_type,
        "switch_time_ms": int(elapsed * 1000),
    }


class LogLevelRequest(BaseModel):
    level: str


@app.post("/log-level")
async def set_log_level(req: LogLevelRequest):
    level = req.level.lower()
    if level not in LOG_LEVELS:
        raise HTTPException(400, detail=f"Unknown level: {req.level}")

    global _current_log_level
    logging.getLogger().setLevel(LOG_LEVELS[level])
    _current_log_level = level
    return {"status": "ok", "level": level}


class PromptRequest(BaseModel):
    name: str
    ref_audio: str
    ref_text: str | None = None
    x_vector_only_mode: bool = False


@app.post("/prompts")
async def create_prompt(req: PromptRequest):
    raw = req.ref_audio
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    audio_bytes = base64.b64decode(raw)

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            engine.create_prompt,
            req.name,
            audio_bytes,
            req.ref_text,
            req.x_vector_only_mode,
        )
    except Exception as e:
        logger.exception("Prompt creation failed")
        raise HTTPException(500, detail=str(e)) from e

    return {"status": "ok", "name": req.name}


@app.get("/prompts")
async def list_prompts():
    names = engine.list_prompts()
    return {"prompts": names, "count": len(names), "max_cached": MAX_CACHED_PROMPTS}


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.ready else "loading",
        "model_loaded": engine.current_task_type,
        "ready": engine.ready and engine.current_task_type is not None,
    }


def render_metrics() -> str:
    """Prometheus exposition body, hand-rolled f-strings like services/restart-count-exporter/ (no prometheus_client dependency)."""
    lines = [
        "# TYPE qwen3_tts_requests_running gauge",
        f"qwen3_tts_requests_running {engine.requests_running}",
        "# TYPE qwen3_tts_requests_waiting gauge",
        f"qwen3_tts_requests_waiting {engine.requests_waiting}",
        "# TYPE qwen3_tts_model_loaded gauge",
    ]
    for task_type in TASK_MODELS:
        value = 1 if task_type == engine.current_task_type else 0
        lines.append(f'qwen3_tts_model_loaded{{task_type="{task_type}"}} {value}')
    return "\n".join(lines) + "\n"


@app.get("/metrics")
async def metrics():
    return Response(
        content=render_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/status")
async def status():
    return {
        "status": "synthesizing" if engine.requests_running > 0 else "idle",
        "requests_running": engine.requests_running,
        "requests_waiting": engine.requests_waiting,
        "model_loaded": engine.current_task_type,
        "log_level": _current_log_level,
    }
