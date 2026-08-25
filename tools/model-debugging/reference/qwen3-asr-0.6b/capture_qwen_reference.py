"""Capture an official Qwen3-ASR-0.6B native batch reference.

The model path must be a complete local snapshot. Offline environment flags
are set before qwen-asr is imported so a missing file fails locally instead of
turning the reference run into an implicit model download.
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
        description="Capture Qwen3-ASR native language/text/timestamp references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-inference-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--language", default=None)
    parser.add_argument("--forced-aligner-dir", type=Path, default=None)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--hash-model-files", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_inventory(model_dir: Path, hash_files: bool) -> list[dict[str, Any]]:
    files = []
    for path in sorted(path for path in model_dir.rglob("*") if path.is_file()):
        entry: dict[str, Any] = {
            "path": path.relative_to(model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        if hash_files:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
    return files


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


def capture_batch(
    model: Any,
    audio_paths: list[Path],
    sample_ids: list[str],
    language: str | None,
    timestamps: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    languages = None if language is None else [language] * len(audio_paths)
    started = time.perf_counter()
    results = model.transcribe(
        audio=[str(path) for path in audio_paths],
        language=languages,
        return_time_stamps=timestamps,
    )
    elapsed_seconds = time.perf_counter() - started
    if not isinstance(results, (list, tuple)):
        results = [results]
    if len(results) != len(audio_paths):
        raise RuntimeError(
            f"Qwen returned {len(results)} results for {len(audio_paths)} inputs"
        )

    rows = []
    for sample_id, path, result in zip(sample_ids, audio_paths, results, strict=True):
        native_language = getattr(result, "language", None)
        text = getattr(result, "text", None)
        timestamps = getattr(result, "time_stamps", None)
        rows.append(
            {
                "sample_id": sample_id,
                "audio_path": str(path),
                "audio_sha256": sha256_file(path),
                "language_requested": language,
                "language_detected": jsonable(native_language),
                "text": jsonable(text),
                "timestamps": jsonable(timestamps),
                "native_result": jsonable(result),
                "batch_size": len(audio_paths),
                "batch_inference_seconds": round(elapsed_seconds, 6),
            }
        )

    batch_info = {
        "sample_ids": sample_ids,
        "batch_size": len(audio_paths),
        "inference_seconds": round(elapsed_seconds, 6),
    }
    return rows, batch_info


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    audio_paths = [path.resolve() for path in args.audio]
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Qwen model directory not found: {model_dir}")
    if not any(model_dir.iterdir()):
        raise FileNotFoundError(f"Qwen model directory is empty: {model_dir}")
    if args.batch_size < 1 or args.max_inference_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    if args.forced_aligner_dir is not None:
        aligner_dir = args.forced_aligner_dir.resolve()
        if not aligner_dir.is_dir():
            raise FileNotFoundError(f"Forced aligner directory not found: {aligner_dir}")
    else:
        aligner_dir = None
    if args.timestamps and aligner_dir is None:
        raise ValueError("--timestamps requires --forced-aligner-dir")
    for audio_path in audio_paths:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio fixture not found: {audio_path}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": args.device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if aligner_dir is not None:
        model_kwargs["forced_aligner"] = str(aligner_dir)
        model_kwargs["forced_aligner_kwargs"] = {
            "dtype": dtype,
            "device_map": args.device_map,
        }

    model = Qwen3ASRModel.from_pretrained(str(model_dir), **model_kwargs)

    rows_by_id: dict[str, dict[str, Any]] = {}
    batches = []
    for batch_index, start in enumerate(range(0, len(audio_paths), args.batch_size)):
        batch_paths = audio_paths[start : start + args.batch_size]
        batch_ids = [
            f"{index:04d}-{path.stem}"
            for index, path in enumerate(batch_paths, start=start)
        ]
        rows, batch_info = capture_batch(
            model,
            batch_paths,
            batch_ids,
            args.language,
            args.timestamps,
        )
        for row in rows:
            row["batch_index"] = batch_index
            rows_by_id[row["sample_id"]] = row
        batch_info["batch_index"] = batch_index
        batches.append(batch_info)

    rows = [
        rows_by_id[f"{index:04d}-{path.stem}"]
        for index, path in enumerate(audio_paths)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "reference_kind": "qwen3-asr-0.6b-native-inference",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_only": True,
        "source": {
            "model_dir": str(model_dir),
            "model_files": model_inventory(model_dir, args.hash_model_files),
            "forced_aligner_dir": str(aligner_dir) if aligner_dir else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "qwen_asr": importlib.metadata.version("qwen-asr"),
            "device_map": args.device_map,
            "dtype": args.dtype,
        },
        "inference": {
            "max_inference_batch_size": args.max_inference_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "language": args.language,
            "timestamps_requested": args.timestamps,
            "batch_size": args.batch_size,
            "batches": batches,
        },
        "samples": rows,
    }
    args.output.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Qwen reference JSON to {args.output}")


if __name__ == "__main__":
    main()
