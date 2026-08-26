"""Capture an offline FunASR SenseVoiceSmall reference.

The model directory must already contain a complete local SenseVoiceSmall
snapshot. No model or tokenizer download is attempted by this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture SenseVoiceSmall FunASR references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="auto", choices=("auto", "zh", "en", "yue", "ja", "ko"))
    parser.add_argument("--use-itn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hash-model-files", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def inventory(model_dir: Path, hash_files: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        row: dict[str, Any] = {
            "path": path.relative_to(model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if hash_files:
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def capture_batch(model: Any, paths: list[Path], language: str, use_itn: bool) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    results = model.generate(
        input=[str(path) for path in paths],
        language=language,
        use_itn=use_itn,
        batch_size=len(paths),
    )
    elapsed = time.perf_counter() - started
    if not isinstance(results, (list, tuple)):
        results = [results]
    if len(results) != len(paths):
        raise RuntimeError(f"FunASR returned {len(results)} results for {len(paths)} inputs")

    rows = []
    for path, result in zip(paths, results, strict=True):
        native = jsonable(result)
        text = native.get("text") if isinstance(native, dict) else None
        rows.append({
            "sample_id": path.stem,
            "audio_path": str(path),
            "audio_sha256": sha256_file(path),
            "language_requested": language,
            "text": text,
            "native_result": native,
            "batch_size": len(paths),
            "batch_inference_seconds": round(elapsed, 6),
        })
    return rows, elapsed


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    audio_paths = [path.resolve() for path in args.audio]
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        raise FileNotFoundError(f"SenseVoice model directory is missing or empty: {model_dir}")
    for path in audio_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audio fixture not found: {path}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"

    from funasr import AutoModel

    model = AutoModel(
        model=str(model_dir),
        device=args.device,
        vad_model=None,
        trust_remote_code=True,
        remote_code=str(model_dir / "model.py"),
    )
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(audio_paths), args.batch_size)):
        batch = audio_paths[start : start + args.batch_size]
        captured, elapsed = capture_batch(model, batch, args.language, args.use_itn)
        for row in captured:
            row["batch_index"] = batch_index
        rows.extend(captured)
        batches.append({
            "batch_index": batch_index,
            "batch_size": len(batch),
            "inference_seconds": round(elapsed, 6),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "reference_kind": "sensevoice-small-funasr-native",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_only": True,
        "source": {"model_dir": str(model_dir), "model_files": inventory(model_dir, args.hash_model_files)},
        "runtime": {
            "python": platform.python_version(),
            "funasr": importlib.metadata.version("funasr"),
            "device": args.device,
        },
        "inference": {
            "language": args.language,
            "use_itn": args.use_itn,
            "batch_size": args.batch_size,
            "batches": batches,
        },
        "samples": rows,
    }
    args.output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote SenseVoice reference JSON to {args.output}")


if __name__ == "__main__":
    main()
