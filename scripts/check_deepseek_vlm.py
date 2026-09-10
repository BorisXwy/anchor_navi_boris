#!/usr/bin/env python3
"""Make one tiny multimodal request to validate the cloud VLM relay config.

Credentials resolve exactly as in navigation runs: OPENROUTER_API_KEY /
OPENROUTER_API_URL / NAVI_OPENROUTER_MODEL from the environment (normally
exported by ``source local_env.sh``), then ``.env.deepseek``, then a direct
scan of ``local_env.sh`` for the same exports.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_harness import DeepSeekBackend, VLMProviderFatalError  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=None,
                        help="optional .env.deepseek-style KEY=VALUE file")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # A deterministic, low-resolution image keeps this connectivity check cheap.
    image = np.zeros((64, 96, 3), np.uint8)
    image[:, :48] = (40, 190, 80)
    image[:, 48:] = (45, 70, 210)
    schema = {
        "type": "object",
        "required": ["left_color", "right_color"],
        "properties": {
            "left_color": {"type": "string"},
            "right_color": {"type": "string"},
        },
    }
    backend = DeepSeekBackend(
        model=args.model, base_url=args.base_url, env_file=args.env_file,
        timeout=90, image_detail="low")
    try:
        result = backend.generate_json(
            "Identify the dominant color on the left and right halves of the image.",
            [image], schema)
    except VLMProviderFatalError as exc:
        print(json.dumps({
            "ok": False,
            "endpoint": backend.endpoint,
            "model": backend.model,
            "credential_source": backend.credential_source,
            "error": str(exc),
            "hint": "run `source local_env.sh` (template: local_env.sh.example)",
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "ok": True,
        "endpoint": backend.endpoint,
        "model": backend.model,
        "credential_source": backend.credential_source,
        "result": result,
        "usage": backend.last_usage,
        "call_meta": backend.last_call_meta,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
