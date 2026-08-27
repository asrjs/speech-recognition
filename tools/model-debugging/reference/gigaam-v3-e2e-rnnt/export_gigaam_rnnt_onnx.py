"""Export official GigaAM v3 E2E RNN-T through `model.to_onnx`.

Produces encoder/decoder/joint graphs plus a piece vocabulary sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

OFFICIAL_MODEL_NAME = "v3_e2e_rnnt"
OFFICIAL_MD5 = "2730de7545ac43ad256485a462b0a27a"
GIGAAM_SRC = Path(r"N:\github\salute-developers\GigaAM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official GigaAM RNN-T ONNX via model.to_onnx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--download-root", type=Path, default=Path(r"N:\models\gigaam\official-cache"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"N:\models\onnx\gigaam\v3-e2e-rnnt"))
    parser.add_argument("--model-name", default=OFFICIAL_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_vocab(tokenizer) -> list[str]:
    if getattr(tokenizer, "charwise", False):
        return list(tokenizer.vocab)
    return [tokenizer.model.IdToPiece(index) for index in range(len(tokenizer.model))]


def write_vocab_file(path: Path, vocab: list[str]) -> None:
    lines = [f"{token} {index}" for index, token in enumerate(vocab)]
    lines.append(f"<blk> {len(vocab)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_info(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    ckpt_path = args.download_root.expanduser() / f"{args.model_name}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Official checkpoint missing: {ckpt_path}")
    if args.model_name == OFFICIAL_MODEL_NAME and md5_file(ckpt_path) != OFFICIAL_MD5:
        raise RuntimeError(f"Checkpoint MD5 mismatch for {ckpt_path}")

    sys.path.insert(0, str(GIGAAM_SRC))
    import gigaam

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = gigaam.load_model(
        args.model_name,
        fp16_encoder=False,
        use_flash=False,
        device=args.device,
        download_root=str(args.download_root),
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    model.to_onnx(dir_path=str(args.output_dir), dtype=dtype)
    vocab = tokenizer_vocab(model.decoding.tokenizer)
    vocab_path = args.output_dir / f"{args.model_name}_vocab.txt"
    write_vocab_file(vocab_path, vocab)
    sidecar = {
        "model_name": args.model_name,
        "export_dtype": args.dtype,
        "checkpoint_md5": md5_file(ckpt_path),
        "blank_id": int(model.decoding.blank_id),
        "vocab_size": len(vocab),
        "pred_hidden": int(model.cfg.head.decoder.pred_hidden),
        "pred_rnn_layers": int(model.cfg.head.decoder.pred_rnn_layers),
        "max_symbols_per_step": int(model.decoding.max_symbols),
        "encoder": file_info(args.output_dir / f"{args.model_name}_encoder.onnx"),
        "decoder": file_info(args.output_dir / f"{args.model_name}_decoder.onnx"),
        "joint": file_info(args.output_dir / f"{args.model_name}_joint.onnx"),
        "yaml": file_info(args.output_dir / f"{args.model_name}.yaml"),
        "vocab": file_info(vocab_path),
        "inputs": {
            "encoder": ["audio_signal", "length"],
            "decoder": ["x", "hi", "ci"],
            "joint": ["enc", "dec"],
        },
        "outputs": {
            "encoder": ["encoded", "encoded_len"],
            "decoder": ["dec", "ho", "co"],
            "joint": ["joint"],
        },
        "opset": 17,
        "status": "experimental-official-export",
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
