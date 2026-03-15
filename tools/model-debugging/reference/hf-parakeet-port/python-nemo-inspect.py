from __future__ import annotations

import argparse
import inspect
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import torch
from nemo.collections.asr.models import ASRModel


def default_audio_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fixtures" / "audio" / "librivox.org.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeMo ASR inference and dump hypothesis/token details for debugging decode parity.",
    )
    parser.add_argument("--audio", type=Path, default=default_audio_path())
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--restore-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
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


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def hypothesis_token_ids(hypothesis: Any) -> list[int]:
    seq = getattr(hypothesis, "y_sequence", None)
    if seq is None:
        return []
    if hasattr(seq, "tolist"):
        return [int(x) for x in seq.tolist()]
    return [int(x) for x in seq]


def decode_pieces(model: Any, token_ids: Iterable[int]) -> list[str] | None:
    tokenizer = getattr(model, "tokenizer", None)
    ids = list(token_ids)
    if tokenizer is None or not ids:
        return None

    if hasattr(tokenizer, "ids_to_tokens"):
        try:
            return list(tokenizer.ids_to_tokens(ids))
        except Exception:  # noqa: BLE001
            pass

    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "id_to_piece"):
        try:
            return [inner.id_to_piece(int(i)) for i in ids]
        except Exception:  # noqa: BLE001
            pass

    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "id_to_piece"):
        try:
            return [inner.id_to_piece(int(i)) for i in ids]
        except Exception:  # noqa: BLE001
            pass

    return None


def load_model(args: argparse.Namespace, device: str) -> Any:
    if args.restore_path:
        model = ASRModel.restore_from(str(args.restore_path), map_location=device)
    else:
        model = ASRModel.from_pretrained(model_name=args.model, map_location=device)
    model = model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def call_transcribe(model: Any, audio_path: Path, args: argparse.Namespace) -> Any:
    kwargs = {
        "batch_size": args.batch_size,
        "return_hypotheses": True,
        "verbose": False,
        "timestamps": args.timestamps,
    }
    try:
        accepted = set(inspect.signature(model.transcribe).parameters)
    except Exception:  # noqa: BLE001
        accepted = set(kwargs)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return model.transcribe([str(audio_path)], **filtered_kwargs)


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = resolve_device(args.device)
    model = load_model(args, device)
    results = call_transcribe(model, audio_path, args)
    if not results:
        raise RuntimeError("NeMo returned no transcription results.")

    hypothesis = results[0]
    token_ids = hypothesis_token_ids(hypothesis)
    token_pieces = decode_pieces(model, token_ids)

    payload = {
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "nemo_toolkit": version("nemo_toolkit"),
            "audio": str(audio_path),
            "model": args.model,
            "restore_path": str(args.restore_path.resolve()) if args.restore_path else None,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "batch_size": args.batch_size,
            "timestamps": bool(args.timestamps),
        },
        "result": {
            "text": getattr(hypothesis, "text", None),
            "normalized_text": normalize_text(getattr(hypothesis, "text", "")),
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "score": normalize_for_json(getattr(hypothesis, "score", None)),
            "timestamp": normalize_for_json(getattr(hypothesis, "timestamp", None)),
            "timestep": normalize_for_json(getattr(hypothesis, "timestep", None)),
            "context_librivox": find_context(getattr(hypothesis, "text", ""), "librivox"),
            "context_domain": find_context(getattr(hypothesis, "text", ""), "librivox.org"),
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
