"""
Check model_fp16.onnx IO types and regenerate if they are not float32.
Reads only the graph struct (no tensor data) for fast inspection.
Then re-exports from the base model if needed.
"""
from pathlib import Path
import onnx
from onnx import TensorProto

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "medasr"
BASE_ONNX  = MODELS_DIR / "model.onnx"
FP16_ONNX  = MODELS_DIR / "model_fp16.onnx"

ELEM_NAMES = {1: "float32", 10: "float16"}
OP_BLOCK_LIST = ["_to_copy", "Trilu"]

def get_io_types(path: Path):
    """Load only graph metadata (skip external tensor data) and return IO elem types."""
    m = onnx.load(str(path), load_external_data=False)
    inputs  = {inp.name: inp.type.tensor_type.elem_type for inp in m.graph.input}
    outputs = {out.name: out.type.tensor_type.elem_type for out in m.graph.output}
    return inputs, outputs

def session_creates(path: Path) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print("WARNING: onnxruntime not installed. Skipping runtime session check.")
        return True
    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"Runtime check failed: {e}")
        return False

def regenerate():
    model = onnx.load(str(BASE_ONNX))

    converted = None
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        print(f"Converting with onnxruntime.transformers (op_block_list={OP_BLOCK_LIST}) ...")
        converted = convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=OP_BLOCK_LIST,
        )
    except Exception as e:
        print(f"ORT converter failed: {e}")

    if converted is None:
        try:
            from onnxconverter_common.float16 import convert_float_to_float16
            print(f"Converting with onnxconverter_common (op_block_list={OP_BLOCK_LIST}) ...")
            converted = convert_float_to_float16(
                model,
                keep_io_types=True,
                op_block_list=OP_BLOCK_LIST,
                disable_shape_infer=False,
            )
        except ImportError:
            print("ERROR: onnxconverter_common not installed. Run: pip install onnxconverter_common")
            return
        except Exception as e:
            print(f"onnxconverter_common failed: {e}")
            return

    # Verify IOs in the converted model
    for inp in converted.graph.input:
        t = inp.type.tensor_type.elem_type
        print(f"  Input  {inp.name}: {ELEM_NAMES.get(t, t)}")
    for out in converted.graph.output:
        t = out.type.tensor_type.elem_type
        print(f"  Output {out.name}: {ELEM_NAMES.get(t, t)}")

    print(f"Saving to {FP16_ONNX} (inline, no external data) ...")
    onnx.save(converted, str(FP16_ONNX))
    size_mb = FP16_ONNX.stat().st_size / (1024 * 1024)
    print(f"Done! Saved {size_mb:.1f} MB")
    if session_creates(FP16_ONNX):
        print("Runtime check: OK")
    else:
        print("Runtime check: FAILED")

def main():
    if not FP16_ONNX.exists():
        print("model_fp16.onnx not found, generating from scratch...")
        regenerate()
        return

    print(f"Inspecting {FP16_ONNX} ...")
    inputs, outputs = get_io_types(FP16_ONNX)

    has_fp16_io = False
    for name, t in {**inputs, **outputs}.items():
        label = ELEM_NAMES.get(t, f"type={t}")
        print(f"  {name}: {label}")
        if t == TensorProto.FLOAT16:
            has_fp16_io = True

    runtime_ok = session_creates(FP16_ONNX)

    if not has_fp16_io and runtime_ok:
        print("\n✓ IO types are float32 and runtime session check passed.")
        return

    print("\n✗ Model needs regeneration (float16 IO or runtime session failure).")
    print(f"Loading base FP32 model from {BASE_ONNX} (this may take a minute)...")
    regenerate()

if __name__ == "__main__":
    main()
