"""Capture official GigaAM multilingual CTC reference outputs.

The oracle is `gigaam.load_model('multilingual_ctc').transcribe`, not a
third-party ONNX conversion. The checkpoint must already exist under
`--download-root`; this script does not invent a substitute artifact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

OFFICIAL_MODEL_NAME = "multilingual_ctc"
OFFICIAL_MD5 = "5379d887c53ccd9cb95981e2a1832720"
OFFICIAL_WEIGHT_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/multilingual_ctc.ckpt"
OFFICIAL_REPO = "https://github.com/salute-developers/GigaAM"
HF_MIRROR = "ai-sage/GigaAM-Multilingual"
HF_REVISION = "ctc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture official GigaAM multilingual CTC references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path(r"N:\models\gigaam\official-cache"),
    )
    parser.add_argument("--model-name", default=OFFICIAL_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-dir", type=Path, default=None)
    parser.add_argument("--include-waveform", action="store_true")
    parser.add_argument("--word-timestamps", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:
        return None


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tensor_stats(array: np.ndarray) -> dict[str, Any]:
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item) and getattr(value, "ndim", 1) == 0:
        return jsonable(value.item())
    return str(value)


def dump_npy(directory: Path | None, sample_id: str, name: str, array: np.ndarray) -> str | None:
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.{name}.npy"
    np.save(path, np.ascontiguousarray(array))
    return str(path)


def inspect_mel(preprocessor: Any) -> dict[str, Any]:
    featurizer = preprocessor.featurizer[0]
    return {
        "class": type(preprocessor).__name__,
        "hop_length": int(preprocessor.hop_length),
        "win_length": int(preprocessor.win_length),
        "n_fft": int(preprocessor.n_fft),
        "center": bool(preprocessor.center),
        "sample_rate": int(getattr(featurizer, "sample_rate", 16000)),
        "n_mels": int(getattr(featurizer, "n_mels", 0)),
        "f_min": float(getattr(featurizer, "f_min", 0.0)),
        "f_max": float(getattr(featurizer, "f_max", 0.0) or 0.0),
        "power": float(getattr(featurizer, "power", 2.0)),
        "norm": str(getattr(featurizer, "norm", None)),
        "mel_scale": str(getattr(featurizer, "mel_scale", None)),
        "window_fn": getattr(getattr(featurizer, "window_fn", None), "__name__", str(getattr(featurizer, "window_fn", None))),
        "normalized": bool(getattr(featurizer, "normalized", False)),
        "pad": int(getattr(featurizer, "pad", 0)),
        "onesided": bool(getattr(featurizer, "onesided", True)),
    }


def write_vocab_file(path: Path, vocab: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{token} {index}" for index, token in enumerate(vocab)]
    lines.append(f"<blk> {len(vocab)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def capture_sample(
    model: Any,
    audio_path: Path,
    tensor_dir: Path | None,
    include_waveform: bool,
    word_timestamps: bool,
) -> dict[str, Any]:
    from gigaam.preprocess import load_audio

    wav = load_audio(str(audio_path))
    wav_batch = wav.unsqueeze(0)
    length = torch.full([1], wav.shape[-1], dtype=torch.long)
    features, feature_lengths = model.preprocessor(wav_batch, length)
    encoded, encoded_len = model.encoder(features, feature_lengths)
    log_probs = model.head(encoded)
    decoded = model.decoding.decode(model.head, encoded, encoded_len)[0]
    text, token_ids, token_frames = decoded
    started = time.perf_counter()
    result = model.transcribe(str(audio_path), word_timestamps=word_timestamps)
    elapsed = time.perf_counter() - started

    features_np = features.detach().cpu().numpy()
    log_probs_np = log_probs.detach().cpu().numpy()
    encoded_np = encoded.detach().cpu().numpy()
    wav_np = wav.detach().cpu().numpy()
    sample_id = audio_path.stem
    argmax_ids = log_probs_np.argmax(axis=-1)[0].astype(np.int32)

    return {
        "sample_id": sample_id,
        "audio": {
            "path": str(audio_path.resolve()),
            "sha256": sha256_file(audio_path),
            "sample_rate": 16000,
            "num_samples": int(wav_np.shape[-1]),
            "duration_seconds": float(wav_np.shape[-1] / 16000),
            "waveform": tensor_stats(wav_np)
            | ({"data": wav_np.astype(np.float32).tolist()} if include_waveform else {}),
            "waveform_npy": dump_npy(tensor_dir, sample_id, "waveform", wav_np.astype(np.float32)),
        },
        "text": str(result),
        "token_ids": [int(value) for value in token_ids],
        "token_frames": [int(value) for value in token_frames],
        "words": [
            {"text": word.text, "start": word.start, "end": word.end}
            for word in (result.words or [])
        ],
        "transcribe_seconds": elapsed,
        "stages": {
            "features": tensor_stats(features_np)
            | {
                "npy": dump_npy(tensor_dir, sample_id, "features", features_np),
                "lengths": [int(value) for value in feature_lengths.tolist()],
            },
            "encoded": tensor_stats(encoded_np)
            | {
                "npy": dump_npy(tensor_dir, sample_id, "encoded", encoded_np),
                "lengths": [int(value) for value in encoded_len.tolist()],
            },
            "log_probs": tensor_stats(log_probs_np)
            | {
                "npy": dump_npy(tensor_dir, sample_id, "log_probs", log_probs_np),
                "argmax_ids": argmax_ids.tolist(),
            },
        },
    }


def main() -> int:
    args = parse_args()
    download_root = args.download_root.expanduser()
    ckpt_path = download_root / f"{args.model_name}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Official checkpoint missing: {ckpt_path}. "
            f"Download {OFFICIAL_WEIGHT_URL} into --download-root first."
        )
    observed_md5 = md5_file(ckpt_path)
    if args.model_name == OFFICIAL_MODEL_NAME and observed_md5 != OFFICIAL_MD5:
        raise RuntimeError(
            f"Checkpoint MD5 {observed_md5} != official {OFFICIAL_MD5}. "
            f"Delete {ckpt_path} and re-download from {OFFICIAL_WEIGHT_URL}."
        )

    sys.path.insert(0, str(Path(r"N:\github\salute-developers\GigaAM")))
    import gigaam
    from omegaconf import OmegaConf

    model = gigaam.load_model(
        args.model_name,
        fp16_encoder=False,
        use_flash=False,
        device=args.device,
        download_root=str(download_root),
    )
    cfg = jsonable(OmegaConf.to_container(model.cfg, resolve=True))
    vocab = list(model.decoding.tokenizer.vocab)
    blank_id = int(model.decoding.blank_id)
    tensor_dir = args.tensor_dir or (args.output.parent / "tensors")
    samples = [
        capture_sample(model, path, tensor_dir, args.include_waveform, args.word_timestamps)
        for path in args.audio
    ]

    vocab_path = args.output.parent / "multilingual_vocab.txt"
    write_vocab_file(vocab_path, vocab)
    yaml_path = args.output.parent / f"{args.model_name}.yaml"
    OmegaConf.save(model.cfg, str(yaml_path))

    payload = {
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "engine": "official-gigaam-pytorch",
        "provenance": {
            "family": "gigaam-ctc",
            "model_name": args.model_name,
            "official_repo": OFFICIAL_REPO,
            "official_weight_url": OFFICIAL_WEIGHT_URL,
            "huggingface_mirror": HF_MIRROR,
            "huggingface_revision": HF_REVISION,
            "gigaam_git_revision": git_revision(Path(r"N:\github\salute-developers\GigaAM")),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_size_bytes": ckpt_path.stat().st_size,
            "checkpoint_md5": observed_md5,
            "checkpoint_sha256": sha256_file(ckpt_path),
            "license": "MIT",
            "status": "experimental-official-reference",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": args.device,
            "torch": torch.__version__,
            "torchaudio": package_version("torchaudio"),
            "onnx": package_version("onnx"),
            "onnxruntime": package_version("onnxruntime"),
            "gigaam": str(Path(gigaam.__file__).resolve()),
            "cuda": torch.cuda.is_available(),
        },
        "preprocessor": inspect_mel(model.preprocessor),
        "model_cfg": cfg,
        "tokenizer": {
            "kind": "character",
            "vocab": vocab,
            "vocab_size": len(vocab),
            "blank_id": blank_id,
            "vocab_path": str(vocab_path),
            "yaml_path": str(yaml_path),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for sample in samples:
        print(f"  {sample['sample_id']}: {sample['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
