#!/usr/bin/env python3

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_harness import (  # noqa: E402
    DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, DeepSeekBackend,
    VLMProviderFatalError, build_vlm_backend,
)


# The backend sends every request through its own proxy-free opener, so the
# transport seam to mock is OpenerDirector.open rather than urllib.request.urlopen.
OPEN = "urllib.request.OpenerDirector.open"
EMPTY_SCHEMA = {"type": "object", "properties": {}}
RELAY_ENV_KEYS = (
    "OPENROUTER_API_KEY", "OPENROUTER_API_URL", "NAVI_OPENROUTER_MODEL",
    "NAVI_LLM_DISABLE_THINKING", "NAVI_OPENROUTER_DISABLE_PROXY",
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_VLM_MODEL",
    "DEEPSEEK_THINKING", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
)


class FakeHTTPResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _response(payload):
    return FakeHTTPResponse(json.dumps(payload).encode("utf-8"))


def _http_error(url, code, reason, body):
    return urllib.error.HTTPError(url, code, reason, {}, io.BytesIO(body))


def _clean_env():
    """Isolate tests from whatever local_env.sh exported into this shell."""
    return patch.dict(os.environ, {key: "" for key in RELAY_ENV_KEYS})


class DeepSeekBackendTest(unittest.TestCase):
    def setUp(self):
        self._env = _clean_env()
        self._env.start()
        self.addCleanup(self._env.stop)

    def _backend(self, **kwargs):
        # local_env_path=None keeps the repository's real local_env.sh out of
        # unit tests so results do not depend on machine-local credentials.
        kwargs.setdefault("local_env_path", None)
        return DeepSeekBackend(**kwargs)

    def test_multimodal_chat_completion_request_and_json_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env.deepseek"
            env_file.write_text(
                "DEEPSEEK_API_KEY=fake-test-key\n"
                "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
                f"DEEPSEEK_VLM_MODEL={DEFAULT_DEEPSEEK_MODEL}\n"
                "DEEPSEEK_THINKING=disabled\n")
            backend = build_vlm_backend(
                "deepseek", model=None, timeout=12,
                deepseek_env_file=env_file)
            backend.local_env_path = None
            image = np.zeros((24, 32, 3), np.uint8)
            schema = {
                "type": "object",
                "required": ["view_index"],
                "properties": {"view_index": {"type": "integer"}},
            }
            response = {
                "choices": [{"message": {"content": "{\"view_index\": 2}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
            with patch(OPEN, return_value=_response(response)) as opened:
                result = backend.generate_json("choose a floor view", [image], schema)

            self.assertEqual(result, {"view_index": 2})
            request = opened.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer fake-test-key")
            self.assertEqual(opened.call_args.kwargs["timeout"], 12)
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], DEFAULT_DEEPSEEK_MODEL)
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["messages"][0]["role"], "user")
            content = payload["messages"][0]["content"]
            self.assertEqual([part["type"] for part in content],
                             ["text", "image_url"])
            self.assertIn("JSON Schema", content[0]["text"])
            self.assertTrue(content[1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"))
            self.assertEqual(content[1]["image_url"]["detail"], "high")
            self.assertEqual(backend.last_usage["completion_tokens"], 4)
            self.assertEqual(backend.last_finish_reason, "stop")
            self.assertEqual(backend.credential_source,
                             f"{env_file}:DEEPSEEK_API_KEY")
            meta = backend.last_call_meta
            self.assertEqual(meta["attempts"], 1)
            self.assertEqual(meta["http_status"], 200)
            self.assertEqual(meta["endpoint"], "https://api.deepseek.com/chat/completions")

    def test_relay_env_names_take_precedence_over_deepseek_names(self):
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "relay-key",
            "OPENROUTER_API_URL": "https://www.dmxapi.cn/v1/chat/completions",
            "NAVI_OPENROUTER_MODEL": "relay-model",
            "NAVI_LLM_DISABLE_THINKING": "1",
            "DEEPSEEK_API_KEY": "legacy-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_VLM_MODEL": "legacy-model",
            "DEEPSEEK_THINKING": "enabled",
        }):
            backend = self._backend()
            response = {"choices": [{"message": {"content": "{}"}}]}
            with patch(OPEN, return_value=_response(response)) as opened:
                backend.generate_json("prompt", [], EMPTY_SCHEMA)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://www.dmxapi.cn/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer relay-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "relay-model")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(backend.credential_source, "environment:OPENROUTER_API_KEY")

    def test_default_base_url_is_the_dmxapi_relay(self):
        backend = self._backend(api_key="k")
        self.assertEqual(DEFAULT_DEEPSEEK_BASE_URL, "https://www.dmxapi.cn/v1")
        self.assertEqual(backend.endpoint, "https://www.dmxapi.cn/v1/chat/completions")

    def test_local_env_sh_fallback_skips_shell_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            local_env = Path(temporary) / "local_env.sh"
            local_env.write_text(
                '# Machine-local runtime environment\n'
                'eval "$(conda shell.bash hook)"\n'
                'conda activate anchor_navi\n'
                'export OPENROUTER_API_KEY="sk-from-local-env"\n'
                'export OPENROUTER_API_URL="https://www.dmxapi.cn/v1"\n'
                'export NAVI_LLM_DISABLE_THINKING=0\n'
                'cd "$ANCHOR_NAVI_ROOT"\n')
            backend = DeepSeekBackend(local_env_path=local_env)
        self.assertEqual(backend.api_key, "sk-from-local-env")
        self.assertEqual(backend.credential_source, f"{local_env}:OPENROUTER_API_KEY")
        self.assertEqual(backend.endpoint, "https://www.dmxapi.cn/v1/chat/completions")
        self.assertEqual(backend.thinking, "enabled")

    def test_missing_key_fails_before_network_call(self):
        backend = self._backend(api_key="")
        with patch(OPEN) as opened:
            with self.assertRaisesRegex(VLMProviderFatalError, "DEEPSEEK_API_KEY"):
                backend.generate_json("prompt", [], EMPTY_SCHEMA)
            opened.assert_not_called()

    def test_placeholder_key_is_rejected_before_network_call(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "your_dmxapi_key_here"}):
            backend = self._backend()
        with patch(OPEN) as opened:
            with self.assertRaisesRegex(VLMProviderFatalError, "OPENROUTER_API_KEY"):
                backend.generate_json("prompt", [], EMPTY_SCHEMA)
            opened.assert_not_called()

    def test_account_http_error_is_provider_fatal(self):
        backend = self._backend(api_key="fake-test-key")
        error = _http_error(backend.base_url, 402, "Payment Required",
                            b'{"error":{"message":"Insufficient Balance"}}')
        with patch(OPEN, side_effect=error), patch("time.sleep") as sleep:
            with self.assertRaisesRegex(VLMProviderFatalError, "DeepSeek HTTP 402"):
                backend.generate_json("prompt", [], EMPTY_SCHEMA)
        sleep.assert_not_called()

    def test_transient_http_error_remains_retryable_runtime_error(self):
        backend = self._backend(api_key="fake-test-key", max_attempts=1)
        error = _http_error(backend.base_url, 429, "Too Many Requests",
                            b'{"error":{"message":"rate limited"}}')
        with patch(OPEN, side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "DeepSeek HTTP 429"):
                backend.generate_json("prompt", [], EMPTY_SCHEMA)

    def test_transient_errors_retry_with_backoff_then_succeed(self):
        backend = self._backend(api_key="fake-test-key", max_attempts=3,
                                retry_base_s=2.0, retry_cap_s=40.0)
        unavailable = _http_error(backend.base_url, 503, "Service Unavailable", b"busy")
        success = _response({"choices": [{"message": {"content": "{\"ok\": true}"},
                                          "finish_reason": "stop"}]})
        with patch(OPEN, side_effect=[unavailable, urllib.error.URLError("reset"),
                                      success]), \
                patch("time.sleep") as sleep, patch("random.random", return_value=0.0):
            result = backend.generate_json("prompt", [], EMPTY_SCHEMA)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(backend.last_call_meta["attempts"], 3)
        self.assertEqual(backend.last_call_meta["http_status"], 200)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    def test_non_retryable_client_error_does_not_retry(self):
        backend = self._backend(api_key="fake-test-key", max_attempts=3)
        error = _http_error(backend.base_url, 400, "Bad Request", b"bad")
        with patch(OPEN, side_effect=error) as opened, patch("time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "DeepSeek HTTP 400"):
                backend.generate_json("prompt", [], EMPTY_SCHEMA)
        self.assertEqual(opened.call_count, 1)
        sleep.assert_not_called()

    def test_proxy_environment_is_ignored_by_default(self):
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:7890",
                                     "https_proxy": "http://127.0.0.1:7890"}):
            backend = self._backend(api_key="k")
            default_proxies = [
                handler.proxies for handler in urllib.request.build_opener().handlers
                if isinstance(handler, urllib.request.ProxyHandler)]
        # Control: a stock opener in that environment would route via 7890;
        # the backend's empty ProxyHandler suppresses it and registers none.
        self.assertEqual(default_proxies[0]["http"], "http://127.0.0.1:7890")
        self.assertTrue(backend.disable_proxy)
        proxies = [handler.proxies for handler in backend._opener.handlers
                   if isinstance(handler, urllib.request.ProxyHandler)]
        self.assertEqual(proxies, [])

    def test_proxy_bypass_can_be_switched_off(self):
        with patch.dict(os.environ, {"NAVI_OPENROUTER_DISABLE_PROXY": "0"}):
            backend = self._backend(api_key="k")
        self.assertFalse(backend.disable_proxy)
        response = {"choices": [{"message": {"content": "{}"}}]}
        with patch("urllib.request.urlopen", return_value=_response(response)) as urlopen:
            backend.generate_json("prompt", [], EMPTY_SCHEMA)
        urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
