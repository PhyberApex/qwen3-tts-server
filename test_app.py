#!/usr/bin/env python3
import logging
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


def _install_heavy_dep_stubs():
    """Exercise app.py against fakes for the GPU-only deps — no CUDA needed."""
    for name in ("numpy", "soundfile", "torch"):
        sys.modules.setdefault(name, MagicMock(name=name))


_install_heavy_dep_stubs()

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class RenderMetrics(unittest.TestCase):
    def setUp(self):
        self.engine = app.TTSEngine()

    def tearDown(self):
        app.engine = self.engine

    def test_has_type_lines_for_all_three_gauges(self):
        app.engine = self.engine
        body = app.render_metrics()
        self.assertIn("# TYPE qwen3_tts_requests_running gauge", body)
        self.assertIn("# TYPE qwen3_tts_requests_waiting gauge", body)
        self.assertIn("# TYPE qwen3_tts_model_loaded gauge", body)

    def test_emits_one_model_loaded_series_per_known_task_type(self):
        app.engine = self.engine
        body = app.render_metrics()
        for task_type in app.TASK_MODELS:
            self.assertIn(f'qwen3_tts_model_loaded{{task_type="{task_type}"}}', body)

    def test_all_model_loaded_series_are_zero_when_nothing_loaded(self):
        app.engine = self.engine
        self.engine.current_task_type = None
        body = app.render_metrics()
        for task_type in app.TASK_MODELS:
            self.assertIn(f'qwen3_tts_model_loaded{{task_type="{task_type}"}} 0', body)

    def test_only_the_loaded_task_type_reads_one(self):
        app.engine = self.engine
        self.engine.current_task_type = "VoiceDesign"
        body = app.render_metrics()
        for task_type in app.TASK_MODELS:
            expected = 1 if task_type == "VoiceDesign" else 0
            self.assertIn(f'qwen3_tts_model_loaded{{task_type="{task_type}"}} {expected}', body)

    def test_switching_models_flips_the_series_rather_than_stacking(self):
        app.engine = self.engine
        self.engine.current_task_type = "CustomVoice"
        first = app.render_metrics()
        self.assertIn('qwen3_tts_model_loaded{task_type="CustomVoice"} 1', first)

        self.engine.current_task_type = "Base"
        second = app.render_metrics()
        self.assertIn('qwen3_tts_model_loaded{task_type="CustomVoice"} 0', second)
        self.assertIn('qwen3_tts_model_loaded{task_type="Base"} 1', second)

    def test_running_and_waiting_counts_are_interpolated(self):
        app.engine = self.engine
        self.engine.requests_running = 3
        self.engine.requests_waiting = 7
        body = app.render_metrics()
        self.assertIn("qwen3_tts_requests_running 3", body)
        self.assertIn("qwen3_tts_requests_waiting 7", body)


class MetricsEndpoint(unittest.TestCase):
    def setUp(self):
        self.original_engine = app.engine
        app.engine = app.TTSEngine()
        app.engine.ready = True
        app.engine.current_task_type = "Base"
        self.client = TestClient(app.app)

    def tearDown(self):
        app.engine = self.original_engine

    def test_serves_prometheus_exposition_format(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("text/plain"))
        self.assertIn("qwen3_tts_requests_running", resp.text)
        self.assertIn('qwen3_tts_model_loaded{task_type="Base"} 1', resp.text)

    def test_metrics_response_is_not_json(self):
        resp = self.client.get("/metrics")
        with self.assertRaises(Exception):
            resp.json()


class StatusEndpoint(unittest.TestCase):
    def setUp(self):
        self.original_engine = app.engine
        app.engine = app.TTSEngine()
        app.engine.ready = True
        app.engine.current_task_type = "Base"
        app.engine.requests_running = 1
        app.engine.requests_waiting = 2
        self.client = TestClient(app.app)

    def tearDown(self):
        app.engine = self.original_engine

    def test_returns_the_same_json_shape_the_old_metrics_path_served(self):
        resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(
            body,
            {
                "status": "synthesizing",
                "requests_running": 1,
                "requests_waiting": 2,
                "model_loaded": "Base",
                "log_level": "info",
            },
        )

    def test_idle_when_nothing_running(self):
        app.engine.requests_running = 0
        resp = self.client.get("/status")
        self.assertEqual(resp.json()["status"], "idle")


class LogLevelEndpoint(unittest.TestCase):
    def setUp(self):
        self.original_engine = app.engine
        self.original_level = app._current_log_level
        self.original_root_level = logging.getLogger().level
        app.engine = app.TTSEngine()
        app.engine.ready = True
        app.engine.current_task_type = "Base"
        self.client = TestClient(app.app)

    def tearDown(self):
        app.engine = self.original_engine
        app._current_log_level = self.original_level
        logging.getLogger().setLevel(self.original_root_level)

    def test_sets_the_root_logger_level_and_echoes_it_back(self):
        resp = self.client.post("/log-level", json={"level": "debug"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok", "level": "debug"})
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_status_reflects_the_level_that_was_set(self):
        self.client.post("/log-level", json={"level": "warning"})
        resp = self.client.get("/status")
        self.assertEqual(resp.json()["log_level"], "warning")

    def test_unknown_level_is_rejected_and_leaves_the_level_unchanged(self):
        self.client.post("/log-level", json={"level": "debug"})
        resp = self.client.post("/log-level", json={"level": "trace"})
        self.assertEqual(resp.status_code, 400)

        status_resp = self.client.get("/status")
        self.assertEqual(status_resp.json()["log_level"], "debug")


class WaitingCounterConcurrency(unittest.TestCase):
    """requests_waiting must only ever move under a lock, and must stay
    non-zero for the whole time a request is genuinely queued behind an
    in-flight synthesis — not just for the instant of the increment."""

    def setUp(self):
        self.engine = app.TTSEngine()
        self.engine.current_task_type = "Base"
        self.engine.model = MagicMock()

    def test_queued_callers_read_as_waiting_while_first_holds_the_lock(self):
        # Two callers must be genuinely queued, not one: a single queued
        # caller can't distinguish this from the naive fix (incrementing
        # under the synthesis lock), since the in-flight request's own
        # count coincidentally also reads 1 for the whole time it runs.
        entered_generate = threading.Event()
        release_generate = threading.Event()

        def slow_generate(*args, **kwargs):
            entered_generate.set()
            release_generate.wait(timeout=2)
            return ([MagicMock()], 16000)

        with patch.object(self.engine, "_generate", side_effect=slow_generate), \
                patch.object(self.engine, "_encode_wav", return_value=b"wav"):
            first = threading.Thread(
                target=self.engine.synthesize, args=("Base", "hello")
            )
            first.start()
            self.assertTrue(entered_generate.wait(timeout=2))

            queued = [
                threading.Thread(target=self.engine.synthesize, args=("Base", f"queued-{i}"))
                for i in range(2)
            ]
            for t in queued:
                t.start()

            # Give the queued calls a moment to register themselves as
            # waiting and block on the synthesis lock behind the first call.
            time.sleep(0.2)
            self.assertEqual(self.engine.requests_waiting, 2)
            self.assertEqual(self.engine.requests_running, 1)

            release_generate.set()
            first.join(timeout=2)
            for t in queued:
                t.join(timeout=2)

        self.assertEqual(self.engine.requests_waiting, 0)
        self.assertEqual(self.engine.requests_running, 0)

    def test_waiting_never_goes_negative_under_concurrent_requests(self):
        with patch.object(self.engine, "_generate", return_value=([MagicMock()], 16000)), \
                patch.object(self.engine, "_encode_wav", return_value=b"wav"):
            threads = [
                threading.Thread(target=self.engine.synthesize, args=("Base", f"msg-{i}"))
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        self.assertEqual(self.engine.requests_waiting, 0)
        self.assertEqual(self.engine.requests_running, 0)


class Configuration(unittest.TestCase):
    def test_model_ids_default_to_the_published_qwen3_checkpoints(self):
        self.assertEqual(app.task_models_from_env({}), app.DEFAULT_TASK_MODELS)

    def test_each_model_id_is_overridable_on_its_own(self):
        models = app.task_models_from_env({"TTS_MODEL_BASE": "someone/finetune"})
        self.assertEqual(models["Base"], "someone/finetune")
        self.assertEqual(models["CustomVoice"], app.DEFAULT_TASK_MODELS["CustomVoice"])
        self.assertEqual(models["VoiceDesign"], app.DEFAULT_TASK_MODELS["VoiceDesign"])

    def test_an_empty_override_falls_back_to_the_default(self):
        models = app.task_models_from_env({"TTS_MODEL_VOICE_DESIGN": ""})
        self.assertEqual(models["VoiceDesign"], app.DEFAULT_TASK_MODELS["VoiceDesign"])

    def test_supported_dtypes_resolve_to_torch_attributes(self):
        for name in app.SUPPORTED_DTYPES:
            self.assertIs(app.resolve_dtype(name), getattr(app.torch, name))

    def test_unknown_dtype_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            app.resolve_dtype("int8")
        self.assertIn("int8", str(ctx.exception))

    def test_model_loads_on_the_configured_device_and_dtype(self):
        engine = app.TTSEngine()
        fake_module = MagicMock()
        with patch.dict(sys.modules, {"qwen_tts": fake_module}), \
                patch.object(app, "DEVICE", "cuda:1"), \
                patch.object(app, "DTYPE_NAME", "float16"):
            engine.load_model("Base")
        kwargs = fake_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], "cuda:1")
        self.assertIs(kwargs["dtype"], app.torch.float16)
        self.assertEqual(engine.current_task_type, "Base")


class ApiKey(unittest.TestCase):
    def setUp(self):
        self.original_engine = app.engine
        app.engine = app.TTSEngine()
        app.engine.ready = True
        app.engine.current_task_type = "Base"
        self.client = TestClient(app.app)

    def tearDown(self):
        app.engine = self.original_engine

    def test_without_a_configured_key_every_route_is_open(self):
        with patch.object(app, "API_KEY", None):
            self.assertEqual(self.client.get("/status").status_code, 200)
            self.assertEqual(self.client.get("/prompts").status_code, 200)

    def test_health_and_metrics_stay_open_with_a_key_configured(self):
        with patch.object(app, "API_KEY", "s3cret"):
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(self.client.get("/metrics").status_code, 200)

    def test_missing_bearer_is_rejected_with_a_challenge(self):
        with patch.object(app, "API_KEY", "s3cret"):
            resp = self.client.get("/status")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.headers["www-authenticate"], "Bearer")
        self.assertIn("bearer", resp.json()["detail"].lower())

    def test_wrong_bearer_is_rejected(self):
        with patch.object(app, "API_KEY", "s3cret"):
            resp = self.client.post(
                "/log-level", json={"level": "debug"},
                headers={"Authorization": "Bearer nope"},
            )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(app._current_log_level, "info")

    def test_matching_bearer_passes_regardless_of_scheme_case(self):
        with patch.object(app, "API_KEY", "s3cret"):
            resp = self.client.get("/status", headers={"Authorization": "bearer s3cret"})
        self.assertEqual(resp.status_code, 200)

    def test_a_non_bearer_scheme_is_rejected_even_with_the_right_secret(self):
        with patch.object(app, "API_KEY", "s3cret"):
            resp = self.client.get("/status", headers={"Authorization": "Basic s3cret"})
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
