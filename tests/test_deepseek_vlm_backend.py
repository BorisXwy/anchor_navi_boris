#!/usr/bin/env python3

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_harness import (  # noqa: E402
    DEFAULT_DEEPSEEK_MODEL, DeepSeekBackend, VLMProviderFatalError,
    build_vlm_backend,
)


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class DeepSeekBackendTest(unittest.TestCase):
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
            image = np.zeros((24, 32, 3), np.uint8)
            schema = {
                "type": "object",
                "required": ["view_index"],
                "properties": {"view_index": {"type": "integer"}},
            }
            response = {
                "choices": [{"message": {"content": "{\"view_index\": 2}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
            with patch("urllib.request.urlopen") as urlopen:
                urlopen.return_value = FakeHTTPResponse(
                    json.dumps(response).encode("utf-8"))
                result = backend.generate_json("choose a floor view", [image], schema)

            self.assertEqual(result, {"view_index": 2})
            request = urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Authorization"), "Bearer fake-test-key")
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 12)
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

    def test_missing_key_fails_before_network_call(self):
        with patch.dict(os.environ, {}, clear=True):
            backend = DeepSeekBackend(api_key="")
            with patch("urllib.request.urlopen") as urlopen:
                with self.assertRaisesRegex(
                        VLMProviderFatalError, "DEEPSEEK_API_KEY"):
                    backend.generate_json(
                        "prompt", [], {"type": "object", "properties": {}})
                urlopen.assert_not_called()

    def test_account_http_error_is_provider_fatal(self):
        backend = DeepSeekBackend(api_key="fake-test-key")
        error = urllib.error.HTTPError(
            backend.base_url, 402, "Payment Required", {},
            io.BytesIO(b'{"error":{"message":"Insufficient Balance"}}'))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(
                    VLMProviderFatalError, "DeepSeek HTTP 402"):
                backend.generate_json(
                    "prompt", [], {"type": "object", "properties": {}})

    def test_transient_http_error_remains_retryable_runtime_error(self):
        backend = DeepSeekBackend(api_key="fake-test-key")
        error = urllib.error.HTTPError(
            backend.base_url, 429, "Too Many Requests", {},
            io.BytesIO(b'{"error":{"message":"rate limited"}}'))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "DeepSeek HTTP 429"):
                backend.generate_json(
                    "prompt", [], {"type": "object", "properties": {}})


if __name__ == "__main__":
    unittest.main()
