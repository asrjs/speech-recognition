"""
Export MedASR to FP16 ONNX.

Strategy (in order of preference):
  1. onnxruntime.transformers.float16  – ORT's own converter; best ORT-web compat.
  2. onnxconverter_common.float16      – community converter with op_block_list fix.

Both keep IO types as float32 so the JS inference code is unchanged.
Post-conversion shape inference is run to fix any residual type-annotation gaps.

Usage:
    conda run -n medasr_env python scripts/export_fp16_native.py [--in-onnx PATH] [--out PATH]
"""

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference, TensorProto

ELEM_NAMES = {
    TensorProto.FLOAT:   "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT32:   "int32",
    TensorProto.INT64:   "int64",
}

# Custom/problematic op types that onnxconverter_common converts poorly.
# Keeping them in float32 avoids graph-level type mismatches in WebGPU.
_ONNXCC_OP_BLOCK = ["_to_copy", "Trilu"]


def print_io_types(model, label=""):
    tag = f" [{label}]" if label else ""
    for inp in model.graph.input:
        t = inp.type.tensor_type.elem_type
        print(f"    INPUT  {inp.name}: {ELEM_NAMES.get(t, t)}{tag}")
    for out in model.graph.output:
        t = out.type.tensor_type.elem_type
        print(f"    OUTPUT {out.name}: {ELEM_NAMES.get(t, t)}{tag}")


def _infer_shapes(model):
    try:
        result = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
        print("  shape_inference: OK")
        return result
    except Exception as e:
        print(f"  shape_inference: WARNING – {e} (continuing)")
        return model


def convert_via_ort_transformers(model):
    """ORT's own float16 converter – best compatibility with ORT-web/WebGPU."""
    from onnxruntime.transformers.float16 import convert_float_to_float16  # noqa: PLC0415
    print(f"  converter: onnxruntime.transformers.float16  (op_block_list={_ONNXCC_OP_BLOCK})")
    return convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=_ONNXCC_OP_BLOCK,
    )


def convert_via_onnxcc(model):
    """onnxconverter_common fallback with custom-op blocking."""
    from onnxconverter_common.float16 import convert_float_to_float16  # noqa: PLC0415
    print(f"  converter: onnxconverter_common  (op_block_list={_ONNXCC_OP_BLOCK})")
    return convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=_ONNXCC_OP_BLOCK,
        disable_shape_infer=False,
    )


def export_fp16(in_onnx: Path, out_onnx: Path):
    if not in_onnx.exists():
        print(f"ERROR: FP32 model not found: {in_onnx}")
        print("  Run export_medasr.py first.")
        return

    print(f"Loading: {in_onnx}  ({in_onnx.stat().st_size / 1e6:.0f} MB on disk)")
    model = onnx.load(str(in_onnx))
    print("  Source IO types:")
    print_io_types(model, "before")

    model_fp16 = None
    for attempt, convert_fn in enumerate([convert_via_ort_transformers, convert_via_onnxcc], 1):
        try:
            print(f"\n[Attempt {attempt}] Converting …")
            model_fp16 = convert_fn(model)
            print("  Conversion: OK")
            break
        except Exception as e:
            print(f"  Conversion FAILED: {e}")
            model_fp16 = None

    if model_fp16 is None:
        print("\nAll converters failed. Aborting.")
        return

    print("\nRunning post-conversion shape inference …")
    model_fp16 = _infer_shapes(model_fp16)

    print("\n  Result IO types:")
    print_io_types(model_fp16, "after")

    print(f"\nSaving → {out_onnx}")
    onnx.save(model_fp16, str(out_onnx))
    size_mb = out_onnx.stat().st_size / 1e6
    print(f"Done! {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _base = Path(__file__).resolve().parent.parent / "models" / "medasr"
    parser.add_argument("--in-onnx", type=Path, default=_base / "model.onnx")
    parser.add_argument("--out",     type=Path, default=_base / "model_fp16.onnx")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_fp16(args.in_onnx, args.out)
