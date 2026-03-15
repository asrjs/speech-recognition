from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import transformers
from scipy.signal import resample_poly
from transformers import pipeline


DEFAULT_MODELS = ["openai/whisper-base"]


def default_audio_path() -> Path:
    candidates = [
        Path(r"N:\JFK.wav"),
        Path(__file__).resolve().parents[1] / "public" / "assets" / "Harvard-L2-1.ogg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default audio file was found. Pass --audio explicitly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Python transformers Whisper ASR pipeline behavior on local audio.",
    )
    parser.add_argument("--audio", type=Path, default=default_audio_path())
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--chunk-length-s", type=float, default=30.0)
    parser.add_argument("--stride-length-s", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Store only summaries and omit the full pipeline outputs from the JSON file.",
    )
    return parser.parse_args()


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after downmix, got shape={audio.shape!r}")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    resampled = resample_poly(audio, target_rate, source_rate)
    return np.asarray(resampled, dtype=np.float32)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    text = str(output.get("text", ""))
    chunks = output.get("chunks") or []
    joined_chunk_text = "".join(str(chunk.get("text", "")) for chunk in chunks)
    return {
        "text_length": len(text),
        "chunk_count": len(chunks),
        "normalized_text": normalize_text(text),
        "normalized_joined_chunk_text": normalize_text(joined_chunk_text),
        "joined_chunk_text_matches_text": (
            normalize_text(text) == normalize_text(joined_chunk_text) if chunks else None
        ),
        "first_chunks": normalize_for_json(chunks[:3]),
        "last_chunks": normalize_for_json(chunks[-3:]) if chunks else [],
    }


def build_cases(audio_seconds: float, chunk_length_s: float, stride_length_s: float) -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = [
        ("plain", {}),
        ("segments_auto", {"return_timestamps": True}),
        ("words_auto", {"return_timestamps": "word"}),
    ]
    if audio_seconds > chunk_length_s:
        cases.extend(
            [
                (
                    "plain_chunked",
                    {
                        "chunk_length_s": chunk_length_s,
                        "stride_length_s": stride_length_s,
                    },
                ),
                (
                    "segments_chunked",
                    {
                        "return_timestamps": True,
                        "chunk_length_s": chunk_length_s,
                        "stride_length_s": stride_length_s,
                    },
                ),
                (
                    "words_chunked",
                    {
                        "return_timestamps": "word",
                        "chunk_length_s": chunk_length_s,
                        "stride_length_s": stride_length_s,
                    },
                ),
            ]
        )
    return cases


def run_case(transcriber, audio: np.ndarray, name: str, options: dict[str, Any], summary_only: bool) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = transcriber(audio, **options)
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "name": name,
            "options": normalize_for_json(options),
            "elapsed_ms": elapsed_ms,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    record = {
        "name": name,
        "options": normalize_for_json(options),
        "elapsed_ms": elapsed_ms,
        "summary": summarize_output(output),
    }
    if not summary_only:
        record["output"] = normalize_for_json(output)
    return record


def run_model(model_id: str, audio_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[python-hf-whisper-inspect] loading model={model_id}", file=sys.stderr)
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=args.device,
    )

    input_audio, input_rate = load_audio(audio_path)
    target_rate = int(getattr(transcriber.feature_extractor, "sampling_rate", input_rate))
    audio = resample_audio(input_audio, input_rate, target_rate)
    audio_seconds = len(audio) / float(target_rate)

    cases = build_cases(audio_seconds, args.chunk_length_s, args.stride_length_s)
    results = []
    for name, options in cases:
        print(f"[python-hf-whisper-inspect] running model={model_id} case={name}", file=sys.stderr)
        results.append(run_case(transcriber, audio, name, options, args.summary_only))

    return {
        "model": model_id,
        "audio": {
            "file": str(audio_path),
            "source_sample_rate": input_rate,
            "model_sample_rate": target_rate,
            "source_seconds": round(len(input_audio) / float(input_rate), 3),
            "model_seconds": round(audio_seconds, 3),
            "sample_count": int(len(audio)),
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    models = args.models or DEFAULT_MODELS
    payload = {
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "audio": str(audio_path),
            "chunk_length_s": args.chunk_length_s,
            "stride_length_s": args.stride_length_s,
            "summary_only": bool(args.summary_only),
        },
        "models": [run_model(model_id, audio_path, args) for model_id in models],
    }

    encoded = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"[python-hf-whisper-inspect] wrote {args.output}", file=sys.stderr)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
