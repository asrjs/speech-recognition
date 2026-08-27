"""Run official sherpa-onnx streaming decode for X-ASR-zh-en.

This is the project-documented inference path
(`OnlineRecognizer.from_transducer`, model_type=zipformer2), not a
third-party ONNX wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official sherpa-onnx X-ASR capture.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"N:\models\x-asr\zh-en\chunk-160ms-model"),
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mono_16k(path: Path, sample_rate: int) -> np.ndarray:
    waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != sample_rate:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=sample_rate)
    return waveform.astype(np.float32)


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    encoder = model_dir / f"encoder-{args.chunk_ms}ms.onnx"
    decoder = model_dir / f"decoder-{args.chunk_ms}ms.onnx"
    joiner = model_dir / f"joiner-{args.chunk_ms}ms.onnx"
    tokens = model_dir / "tokens.txt"
    for path in (encoder, decoder, joiner, tokens, args.audio):
        if not path.is_file():
            raise FileNotFoundError(path)

    import sherpa_onnx

    waveform = load_mono_16k(args.audio, args.sample_rate)
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens),
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        num_threads=1,
        sample_rate=args.sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=args.provider,
        model_type="zipformer2",
        enable_endpoint_detection=False,
    )
    stream = recognizer.create_stream()
    started = time.perf_counter()
    # True streaming: feed the official chunk size, then decode while ready.
    chunk = int(args.sample_rate * args.chunk_ms / 1000)
    partials: list[dict[str, object]] = []
    offset = 0
    while offset < waveform.size:
        end = min(waveform.size, offset + chunk)
        stream.accept_waveform(args.sample_rate, waveform[offset:end])
        offset = end
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
            text = recognizer.get_result(stream)
            if not partials or partials[-1]["text"] != text:
                partials.append({"offset_samples": offset, "text": text})
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    final = recognizer.get_result(stream)
    elapsed = time.perf_counter() - started
    duration = waveform.size / args.sample_rate
    payload = {
        "schema_version": 1,
        "reference_kind": "x-asr-zh-en-sherpa-onnx",
        "streaming": "true-stateful-zipformer2",
        "chunk_ms": args.chunk_ms,
        "oracle": "sherpa_onnx.OnlineRecognizer.from_transducer",
        "provider": args.provider,
        "audio": {
            "path": str(args.audio.resolve()),
            "sha256": sha256_file(args.audio),
            "samples": int(waveform.size),
            "duration_sec": duration,
        },
        "artifacts": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in (("encoder", encoder), ("decoder", decoder), ("joiner", joiner), ("tokens", tokens))
        },
        "text": final,
        "partials": partials,
        "metrics": {
            "inference_seconds": round(elapsed, 6),
            "rtf": round(elapsed / duration, 6) if duration else None,
            "partial_updates": len(partials),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(final)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
