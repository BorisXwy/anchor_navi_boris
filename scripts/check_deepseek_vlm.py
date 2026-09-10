#!/usr/bin/env python3
"""Make one tiny multimodal request to validate the local DeepSeek config."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vlm_harness import DeepSeekBackend  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.deepseek")
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
    result = backend.generate_json(
        "Identify the dominant color on the left and right halves of the image.",
        [image], schema)
    print(json.dumps({
        "ok": True,
        "model": backend.model,
        "result": result,
        "usage": backend.last_usage,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

