from __future__ import annotations

import argparse
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

import onnx_asr
import onnxruntime as ort


def default_audio_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fixtures" / "audio" / "librivox.org.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run onnx-asr against a local NeMo/Parakeet ONNX export and dump text/token details.",
    )
    parser.add_argument("--audio", type=Path, default=default_audio_path())
    parser.add_argument(
        "--model-type",
        default="nemo-conformer-tdt",
        help="onnx-asr model type. Use nemo-conformer-tdt for local Parakeet TDT exports.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"N:\models\onnx\nemo\parakeet-tdt-0.6b-v2-onnx"),
        help="Directory containing encoder-model*, decoder_joint-model*, vocab.txt, and config.json.",
    )
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def find_context(text: str, needle: str, radius: int = 64) -> str | None:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return None
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    return text[start:end]


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    model_path = args.model_path.resolve()

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = onnx_asr.load_model(
        args.model_type,
        model_path,
        quantization=args.quantization,
    ).with_timestamps()
    result = model.recognize(audio_path)

    payload = {
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnx_asr": version("onnx-asr"),
            "onnxruntime": ort.__version__,
            "audio": str(audio_path),
            "model_type": args.model_type,
            "model_path": str(model_path),
            "quantization": args.quantization,
        },
        "result": {
            "text": result.text,
            "normalized_text": normalize_text(result.text),
            "token_count": len(result.tokens or []),
            "tokens": normalize_for_json(result.tokens),
            "timestamps": normalize_for_json(result.timestamps),
            "logprobs": normalize_for_json(result.logprobs),
            "context_librivox": find_context(result.text, "librivox"),
            "context_domain": find_context(result.text, "librivox.org"),
        },
    }

    encoded = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
