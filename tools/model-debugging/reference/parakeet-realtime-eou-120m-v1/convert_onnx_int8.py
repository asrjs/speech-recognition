"""
Create INT8 dynamically quantized ONNX variants for Parakeet realtime encoder/decoder exports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Parakeet realtime ONNX exports to INT8.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=["encoder-model.onnx", "decoder_joint-model.onnx"],
        help="Base FP32 ONNX artifacts to quantize.",
    )
    parser.add_argument(
        "--weight-type",
        choices=["qint8", "quint8"],
        default="quint8",
        help="ONNX Runtime dynamic quantization weight type.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization when supported.",
    )
    return parser.parse_args()


def resolve_quant_type(name: str) -> QuantType:
    return QuantType.QInt8 if name == "qint8" else QuantType.QUInt8


def convert_model(
    source_path: Path,
    output_path: Path,
    weight_type: QuantType,
    per_channel: bool,
) -> None:
    print(f"Quantizing {source_path}")
    quantize_dynamic(
        model_input=str(source_path),
        model_output=str(output_path),
        weight_type=weight_type,
        per_channel=per_channel,
    )
    source_size_mb = source_path.stat().st_size / (1024 * 1024)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_path}")
    print(f"Size: {source_size_mb:.2f} MB -> {output_size_mb:.2f} MB")


def main() -> None:
    args = parse_args()
    weight_type = resolve_quant_type(args.weight_type)
    for name in args.artifacts:
        source_path = args.model_dir / name
        if not source_path.exists():
            print(f"Skipping missing artifact: {source_path}")
            continue
        if source_path.name.endswith(".int8.onnx"):
            print(f"Skipping already-INT8 artifact: {source_path}")
            continue
        output_name = f"{source_path.stem}.int8{source_path.suffix}"
        convert_model(source_path, args.model_dir / output_name, weight_type, args.per_channel)


if __name__ == "__main__":
    main()
