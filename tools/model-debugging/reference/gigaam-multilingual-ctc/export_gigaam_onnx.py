"""Export official GigaAM multilingual CTC through `model.to_onnx`.

The exporter loads the official checkpoint and uses the upstream converter.
It does not start from a third-party ONNX file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

OFFICIAL_MODEL_NAME = "multilingual_ctc"
OFFICIAL_MD5 = "5379d887c53ccd9cb95981e2a1832720"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official GigaAM CTC ONNX via model.to_onnx.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path(r"N:\models\gigaam\official-cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"N:\models\onnx\gigaam\multilingual-ctc"),
    )
    parser.add_argument("--model-name", default=OFFICIAL_MODEL_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
        help="ONNX export dtype. Native/WASM parity starts with float32.",
    )
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


def write_vocab_file(path: Path, vocab: list[str]) -> None:
    lines = [f"{token} {index}" for index, token in enumerate(vocab)]
    lines.append(f"<blk> {len(vocab)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ckpt_path = args.download_root.expanduser() / f"{args.model_name}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Official checkpoint missing: {ckpt_path}")
    if args.model_name == OFFICIAL_MODEL_NAME and md5_file(ckpt_path) != OFFICIAL_MD5:
        raise RuntimeError(f"Checkpoint MD5 mismatch for {ckpt_path}")

    sys.path.insert(0, str(Path(r"N:\github\salute-developers\GigaAM")))
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
    vocab = list(model.decoding.tokenizer.vocab)
    write_vocab_file(args.output_dir / "multilingual_vocab.txt", vocab)

    onnx_path = args.output_dir / f"{args.model_name}.onnx"
    yaml_path = args.output_dir / f"{args.model_name}.yaml"
    sidecar = {
        "model_name": args.model_name,
        "export_dtype": args.dtype,
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size if onnx_path.is_file() else None,
        "onnx_sha256": sha256_file(onnx_path) if onnx_path.is_file() else None,
        "yaml_path": str(yaml_path),
        "vocab_path": str(args.output_dir / "multilingual_vocab.txt"),
        "inputs": ["features", "feature_lengths"],
        "outputs": ["log_probs", "encoded_lengths"],
        "blank_id": int(model.decoding.blank_id),
        "vocab_size": len(vocab),
        "checkpoint_md5": md5_file(ckpt_path),
        "opset": 17,
        "status": "experimental-official-export",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sidecar, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
