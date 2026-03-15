"""
Convert exported Canary ONNX artifacts to FP16 while keeping IO dtypes stable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


OP_BLOCK_LIST = ["Trilu"]


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception as error:  # noqa: BLE001
        print(f"shape_inference warning: {error}")
        return model


def convert_model(source_path: Path, output_path: Path) -> None:
    print(f"Loading {source_path}")
    model = onnx.load(str(source_path))

    converted = None
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        converted = convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=OP_BLOCK_LIST,
        )
        print("Converted with onnxruntime.transformers.float16")
    except Exception as ort_error:  # noqa: BLE001
        print(f"ORT float16 converter failed: {ort_error}")

    if converted is None:
        try:
            from onnxconverter_common.float16 import convert_float_to_float16
        except ImportError as error:
            raise RuntimeError(
                "FP16 conversion fallback requires onnxconverter_common. "
                "Install it in the active environment or rely on the ORT float16 converter."
            ) from error

        converted = convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=OP_BLOCK_LIST,
            disable_shape_infer=False,
        )
        print("Converted with onnxconverter_common.float16")

    converted = infer_shapes(converted)
    onnx.save(converted, str(output_path))
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Canary ONNX exports to FP16.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=["encoder-model.onnx", "decoder-model.onnx", "nemo128.onnx"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.artifacts:
        source_path = args.model_dir / name
        if not source_path.exists():
            print(f"Skipping missing artifact: {source_path}")
            continue
        if source_path.name.endswith(".fp16.onnx"):
            print(f"Skipping already-FP16 artifact: {source_path}")
            continue
        output_name = f"{source_path.stem}.fp16{source_path.suffix}"
        convert_model(source_path, args.model_dir / output_name)


if __name__ == "__main__":
    main()
