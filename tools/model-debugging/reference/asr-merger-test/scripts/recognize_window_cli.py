#!/usr/bin/env python3
"""
CLI to request ASR for a single time window. Uses backend if ASR_BACKEND_URL is set,
otherwise local onnx-asr. Results are cached by (audio, start_sec, end_sec).

Example:
  python scripts/recognize_window_cli.py --audio test-data/out.wav --start 0.8 --end 8.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from asr_window_client import recognize_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR on a time window; cache by (audio, start, end).")
    parser.add_argument("--audio", type=Path, required=True, help="Input WAV or m4a.")
    parser.add_argument("--start", type=float, required=True, help="Window start (seconds).")
    parser.add_argument("--end", type=float, required=True, help="Window end (seconds).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory (default: simulation_inputs/asr_cache).")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    parser.add_argument("--no-backend", action="store_true", help="Do not use ASR_BACKEND_URL; use local model only.")
    args = parser.parse_args()

    if args.start >= args.end:
        print("Error: --start must be less than --end", file=sys.stderr)
        sys.exit(1)

    cache_dir = args.cache_dir or _PROJECT_ROOT / "simulation_inputs" / "asr_cache"
    result = recognize_window(
        args.audio,
        args.start,
        args.end,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
        use_backend=False if args.no_backend else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
