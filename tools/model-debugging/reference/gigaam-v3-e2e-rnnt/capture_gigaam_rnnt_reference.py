"""Capture official GigaAM v3 E2E RNN-T PyTorch oracle outputs.

Oracle is `gigaam.load_model('v3_e2e_rnnt').transcribe`, not a third-party ONNX file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

OFFICIAL_MODEL_NAME = "v3_e2e_rnnt"
OFFICIAL_MD5 = "2730de7545ac43ad256485a462b0a27a"
OFFICIAL_WEIGHT_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt.ckpt"
OFFICIAL_TOKENIZER_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt_tokenizer.model"
OFFICIAL_REPO = "https://github.com/salute-developers/GigaAM"
GIGAAM_SRC = Path(r"N:\github\salute-developers\GigaAM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture official GigaAM v3 E2E RNN-T references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, default=Path(r"N:\models\gigaam\official-cache"))
    parser.add_argument("--model-name", default=OFFICIAL_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-dir", type=Path, default=None)
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


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
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
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def dump_npy(directory: Path | None, sample_id: str, name: str, array: np.ndarray) -> str | None:
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sample_id}.{name}.npy"
    np.save(path, np.ascontiguousarray(array))
    return str(path)


def tokenizer_vocab(tokenizer: Any) -> list[str]:
    if getattr(tokenizer, "charwise", False):
        return list(tokenizer.vocab)
    return [tokenizer.model.IdToPiece(index) for index in range(len(tokenizer.model))]


def piece_join(vocab: list[str], token_ids: list[int]) -> str:
    text = "".join(vocab[index] for index in token_ids if 0 <= index < len(vocab))
    return text.replace("\u2581", " ").replace("▁", " ").strip()


def write_vocab_file(path: Path, vocab: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{token} {index}" for index, token in enumerate(vocab)]
    lines.append(f"<blk> {len(vocab)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "window_fn": getattr(
            getattr(featurizer, "window_fn", None),
            "__name__",
            str(getattr(featurizer, "window_fn", None)),
        ),
    }


def capture_sample(model: Any, audio_path: Path, tensor_dir: Path | None, vocab: list[str]) -> dict[str, Any]:
    from gigaam.preprocess import load_audio

    wav = load_audio(str(audio_path))
    wav_batch = wav.unsqueeze(0)
    length = torch.full([1], wav.shape[-1], dtype=torch.long)
    features, feature_lengths = model.preprocessor(wav_batch, length)
    encoded, encoded_len = model.encoder(features, feature_lengths)
    decoded = model.decoding.decode(model.head, encoded, encoded_len)[0]
    text, token_ids, token_frames = decoded
    started = time.perf_counter()
    result = model.transcribe(str(audio_path))
    elapsed = time.perf_counter() - started
    features_np = features.detach().cpu().numpy()
    encoded_np = encoded.detach().cpu().numpy()
    wav_np = wav.detach().cpu().numpy()
    sample_id = audio_path.stem
    ids = [int(value) for value in token_ids]
    return {
        "sample_id": sample_id,
        "audio": {
            "path": str(audio_path.resolve()),
            "sha256": sha256_file(audio_path),
            "sample_rate": 16000,
            "num_samples": int(wav_np.shape[-1]),
            "duration_seconds": float(wav_np.shape[-1] / 16000),
            "waveform_npy": dump_npy(tensor_dir, sample_id, "waveform", wav_np.astype(np.float32)),
        },
        "text": str(result),
        "decode_text": text,
        "piece_join_text": piece_join(vocab, ids),
        "token_ids": ids,
        "token_frames": [int(value) for value in token_frames],
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
        },
    }


def main() -> int:
    args = parse_args()
    download_root = args.download_root.expanduser()
    ckpt_path = download_root / f"{args.model_name}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Official checkpoint missing: {ckpt_path}")
    observed_md5 = md5_file(ckpt_path)
    if args.model_name == OFFICIAL_MODEL_NAME and observed_md5 != OFFICIAL_MD5:
        raise RuntimeError(f"Checkpoint MD5 {observed_md5} != official {OFFICIAL_MD5}")

    sys.path.insert(0, str(GIGAAM_SRC))
    import gigaam
    from omegaconf import OmegaConf

    model = gigaam.load_model(
        args.model_name,
        fp16_encoder=False,
        use_flash=False,
        device=args.device,
        download_root=str(download_root),
    )
    vocab = tokenizer_vocab(model.decoding.tokenizer)
    blank_id = int(model.decoding.blank_id)
    tensor_dir = args.tensor_dir or (args.output.parent / "tensors")
    samples = [capture_sample(model, path, tensor_dir, vocab) for path in args.audio]
    vocab_path = args.output.parent / "v3_e2e_rnnt_vocab.txt"
    write_vocab_file(vocab_path, vocab)
    OmegaConf.save(model.cfg, str(args.output.parent / f"{args.model_name}.yaml"))
    decoder_cfg = model.cfg.head.decoder
    payload = {
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "engine": "official-gigaam-pytorch",
        "provenance": {
            "family": "gigaam-rnnt",
            "model_name": args.model_name,
            "official_repo": OFFICIAL_REPO,
            "official_weight_url": OFFICIAL_WEIGHT_URL,
            "official_tokenizer_url": OFFICIAL_TOKENIZER_URL,
            "gigaam_git_revision": git_revision(GIGAAM_SRC),
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
            "gigaam": str(Path(gigaam.__file__).resolve()),
        },
        "preprocessor": inspect_mel(model.preprocessor),
        "head": {
            "pred_hidden": int(decoder_cfg.pred_hidden),
            "pred_rnn_layers": int(decoder_cfg.pred_rnn_layers),
            "enc_hidden": int(model.cfg.head.joint.enc_hidden),
            "joint_hidden": int(model.cfg.head.joint.joint_hidden),
            "num_classes": int(model.cfg.head.joint.num_classes),
            "max_symbols_per_step": int(model.decoding.max_symbols),
        },
        "tokenizer": {
            "kind": "sentencepiece" if not model.decoding.tokenizer.charwise else "character",
            "blank_id": blank_id,
            "vocab_size": len(vocab),
            "vocab_path": str(vocab_path),
            "piece_join_matches_official": all(
                sample["piece_join_text"] == sample["text"] for sample in samples
            ),
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "text": samples[0]["text"] if samples else None}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
