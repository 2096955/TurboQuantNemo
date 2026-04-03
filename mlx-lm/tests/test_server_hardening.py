import io
import json
import unittest
from queue import Queue
from types import SimpleNamespace

from mlx_lm.server import APIHandler, ResponseGenerator


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
        self.assertIn("worker died", payload["generation_worker_error"])

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


if __name__ == "__main__":
    unittest.main()
