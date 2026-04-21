import io
import json
import unittest
from queue import Queue
from types import SimpleNamespace

from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    ModelDescription,
    ResponseGenerator,
    SamplingArguments,
    _deserialize_shared_payload,
    _serialize_shared_payload,
)


class DummyThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


def make_cli_args(**overrides):
    defaults = {
        "api_key": None,
        "cors_allow_origin": None,
        "max_request_body_bytes": 1024,
        "num_draft_tokens": 3,
        "max_tokens": 16,
        "temp": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "timeout": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_handler(
    *,
    path="/metrics",
    headers=None,
    cli_args=None,
    worker_alive=True,
    worker_error=None,
):
    cli_args = cli_args or make_cli_args()
    model_provider = SimpleNamespace(
        cli_args=cli_args,
        model=object(),
        tokenizer=object(),
    )
    response_generator = SimpleNamespace(
        cli_args=cli_args,
        model_provider=model_provider,
        worker_alive=worker_alive,
        _worker_error=worker_error,
        requests=Queue(maxsize=2),
        metric_requests_total=3,
        metric_prompt_tokens_total=128,
        metric_completion_tokens_total=64,
        metric_duration_seconds_total=1.5,
    )

    handler = object.__new__(APIHandler)
    handler.created = 0
    handler.response_generator = response_generator
    handler.system_fingerprint = "fp"
    handler.headers = headers or {}
    handler.path = path
    handler.wfile = io.BytesIO()
    handler.sent_headers = []
    handler.status_code = None
    handler.send_response = lambda code: setattr(handler, "status_code", code)
    handler.send_header = lambda key, value: handler.sent_headers.append((key, value))
    handler.end_headers = lambda: None
    return handler


class TestServerHardening(unittest.TestCase):
    def test_distributed_payload_roundtrip_uses_explicit_schema(self):
        payload = (
            CompletionRequest(
                request_type="chat",
                prompt="ignored",
                messages=[
                    {"role": "system", "content": "hi"},
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                ],
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                role_mapping={"user": "human"},
            ),
            GenerationArguments(
                model=ModelDescription(
                    model="default_model",
                    draft="default_model",
                    adapter=None,
                ),
                sampling=SamplingArguments(
                    temperature=0.1,
                    top_p=0.9,
                    top_k=20,
                    min_p=0.05,
                    xtc_probability=0.0,
                    xtc_threshold=0.0,
                ),
                logits=LogitsProcessorArguments(
                    logit_bias={12: -1.5, 99: 2.0},
                    repetition_penalty=1.1,
                    repetition_context_size=32,
                ),
                stop_words=["</s>", "DONE"],
                max_tokens=64,
                num_draft_tokens=4,
                logprobs=True,
                top_logprobs=3,
                seed=7,
                chat_template_kwargs={"mode": "strict", "nested": {"x": [1, 2, 3]}},
            ),
        )

        encoded = _serialize_shared_payload(payload)
        decoded = _deserialize_shared_payload(encoded)

        self.assertEqual(decoded, payload)

    def test_generate_rejects_full_queue_immediately(self):
        response_generator = object.__new__(ResponseGenerator)
        response_generator._worker_error = None
        response_generator._generation_thread = DummyThread(alive=True)
        response_generator._queue_put_timeout = 0.0
        response_generator.requests = Queue(maxsize=1)
        response_generator.requests.put(("occupied",))

        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            response_generator.generate(None, None)

    def test_ready_check_requires_live_worker(self):
        handler = make_handler(
            path="/ready",
            worker_alive=False,
            worker_error=RuntimeError("worker died"),
        )

        handler.handle_ready_check()

        self.assertEqual(handler.status_code, 503)
        payload = json.loads(handler.wfile.getvalue().decode())
        self.assertFalse(payload["ready"])
        self.assertFalse(payload["generation_worker_alive"])
        self.assertTrue(payload["generation_worker_error"])

    def test_protected_get_requires_api_key(self):
        handler = make_handler(
            path="/metrics",
            cli_args=make_cli_args(api_key="secret"),
        )

        handler.do_GET()

        self.assertEqual(handler.status_code, 401)
        payload = json.loads(handler.wfile.getvalue().decode())
        self.assertEqual(payload["error"], "Unauthorized")

    def test_health_get_is_unauthenticated(self):
        handler = make_handler(
            path="/health",
            cli_args=make_cli_args(api_key="secret"),
        )

        handler.do_GET()

        self.assertEqual(handler.status_code, 200)
        payload = json.loads(handler.wfile.getvalue().decode())
        self.assertEqual(payload["status"], "ok")

    def test_post_rejects_invalid_stream_options(self):
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": [],
        }
        payload = json.dumps(body).encode()
        handler = make_handler(
            path="/v1/chat/completions",
            headers={"Content-Length": str(len(payload))},
        )
        handler.rfile = io.BytesIO(payload)
        handler.handle_completion = lambda request, stop_words: self.fail(
            "handle_completion should not be reached"
        )

        handler.do_POST()

        self.assertEqual(handler.status_code, 400)
        error = json.loads(handler.wfile.getvalue().decode())["error"]
        self.assertIn("stream_options must be an object", error)

    def test_post_rejects_oversized_body(self):
        payload = b"x" * 20
        handler = make_handler(
            path="/v1/completions",
            headers={"Content-Length": str(len(payload))},
            cli_args=make_cli_args(max_request_body_bytes=8),
        )
        handler.rfile = io.BytesIO(payload)

        handler.do_POST()

        self.assertEqual(handler.status_code, 413)
        error = json.loads(handler.wfile.getvalue().decode())["error"]
        self.assertIn("Request body too large", error)

    def test_cors_is_disabled_by_default(self):
        handler = make_handler(headers={"Origin": "https://example.com"})

        handler._set_completion_headers(200)

        self.assertNotIn(
            ("Access-Control-Allow-Origin", "https://example.com"),
            handler.sent_headers,
        )

    def test_cors_allows_configured_origin(self):
        handler = make_handler(
            headers={"Origin": "https://allowed.example"},
            cli_args=make_cli_args(
                cors_allow_origin="https://allowed.example,https://other.example"
            ),
        )

        handler._set_completion_headers(200)

        self.assertIn(
            ("Access-Control-Allow-Origin", "https://allowed.example"),
            handler.sent_headers,
        )
        self.assertIn(("Vary", "Origin"), handler.sent_headers)

    def test_metrics_json_includes_counters_when_requested(self):
        handler = make_handler(headers={"Accept": "application/json"})

        handler.handle_metrics_request()

        self.assertEqual(handler.status_code, 200)
        payload = json.loads(handler.wfile.getvalue().decode())
        self.assertEqual(payload["queue_depth"], 0)
        self.assertEqual(payload["counters"]["requests_total"], 3)
        self.assertEqual(payload["counters"]["prompt_tokens_total"], 128)
        self.assertIn(("Content-type", "application/json"), handler.sent_headers)

    def test_metrics_prometheus_format_is_default(self):
        handler = make_handler(headers={})

        handler.handle_metrics_request()

        body = handler.wfile.getvalue().decode()
        self.assertIn("# HELP mlx_lm_requests_total", body)
        self.assertIn("mlx_lm_requests_total 3", body)
        self.assertIn("mlx_lm_queue_depth 0", body)
        self.assertIn(
            ("Content-type", "text/plain; version=0.0.4; charset=utf-8"),
            handler.sent_headers,
        )

    def test_post_accepts_timeout_override(self):
        body = {
            "prompt": "hi",
            "max_tokens": 4,
            "timeout": 1.25,
        }
        payload = json.dumps(body).encode()
        handler = make_handler(
            path="/v1/completions",
            headers={"Content-Length": str(len(payload))},
        )
        handler.rfile = io.BytesIO(payload)
        captured = {}

        def fake_handle_completion(request, stop_words):
            captured["timeout"] = handler.timeout

        handler.handle_completion = fake_handle_completion

        handler.do_POST()

        self.assertEqual(captured["timeout"], 1.25)

    def test_post_rejects_negative_timeout(self):
        body = {
            "prompt": "hi",
            "max_tokens": 4,
            "timeout": -1,
        }
        payload = json.dumps(body).encode()
        handler = make_handler(
            path="/v1/completions",
            headers={"Content-Length": str(len(payload))},
        )
        handler.rfile = io.BytesIO(payload)
        handler.handle_completion = lambda request, stop_words: self.fail(
            "handle_completion should not be reached"
        )

        handler.do_POST()

        self.assertEqual(handler.status_code, 400)
        error = json.loads(handler.wfile.getvalue().decode())["error"]
        self.assertIn("timeout must be a non-negative number", error)

    def test_stream_disconnect_calls_ctx_stop(self):
        handler = make_handler(path="/v1/completions", headers={})
        handler.stream = True
        handler.timeout = None
        handler.logit_bias = None
        handler.repetition_penalty = None
        handler.repetition_context_size = 20
        handler.max_tokens = 10
        handler.num_draft_tokens = 0
        handler.logprobs = False
        handler.top_logprobs = 0
        handler.seed = None
        handler.chat_template_kwargs = None
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.top_k = -1
        handler.min_p = 0.0
        handler.xtc_probability = 0.0
        handler.xtc_threshold = 0.0
        handler.requested_model = "test"
        handler.requested_draft_model = None
        handler.adapter = None
        handler.request_id = "test-id"
        handler.object_type = "chat.completion"
        handler.created = 12345
        handler.system_fingerprint = "fp"

        ctx = SimpleNamespace(
            stop=lambda: setattr(ctx, "stopped", True),
            prompt=[1],
            has_thinking=False,
            has_tool_calling=False,
            prompt_cache_count=0,
            eos_token_ids=set(),
            stop_token_sequences=[]
        )
        ctx.stopped = False

        def fake_generate(*args, **kwargs):
            yield SimpleNamespace(text="test", finish_reason=None, token=1, logprob=0.0)
            raise BrokenPipeError()

        handler.response_generator.generate = lambda *args, **kwargs: (ctx, fake_generate())

        handler.handle_completion(CompletionRequest("", "", [], None, None), [])

        self.assertTrue(ctx.stopped)

    def test_generation_timeout_calls_ctx_stop(self):
        handler = make_handler(path="/v1/completions", headers={})
        handler.stream = False
        handler.timeout = 0.001
        handler.logit_bias = None
        handler.repetition_penalty = None
        handler.repetition_context_size = 20
        handler.max_tokens = 10
        handler.num_draft_tokens = 0
        handler.logprobs = False
        handler.top_logprobs = 0
        handler.seed = None
        handler.chat_template_kwargs = None
        handler.temperature = 0.0
        handler.top_p = 1.0
        handler.top_k = -1
        handler.min_p = 0.0
        handler.xtc_probability = 0.0
        handler.xtc_threshold = 0.0
        handler.requested_model = "test"
        handler.requested_draft_model = None
        handler.adapter = None
        handler.request_id = "test-id"
        handler.object_type = "chat.completion"
        handler.created = 12345
        handler.system_fingerprint = "fp"

        ctx = SimpleNamespace(
            stop=lambda: setattr(ctx, "stopped", True),
            prompt=[1],
            has_thinking=False,
            has_tool_calling=False,
            prompt_cache_count=0,
            eos_token_ids=set(),
            stop_token_sequences=[]
        )
        ctx.stopped = False

        import time
        def fake_generate(*args, **kwargs):
            time.sleep(0.01) # exceed timeout
            yield SimpleNamespace(text="test", finish_reason=None, token=1, logprob=0.0)

        handler.response_generator.generate = lambda *args, **kwargs: (ctx, fake_generate())

        handler.handle_completion(CompletionRequest("", "", [], None, None), [])

        self.assertTrue(ctx.stopped)


if __name__ == "__main__":
    unittest.main()
