"""
Validate model_fp16.onnx using ORT Python (CPU EP).
Reports what ORT actually expects for inputs/outputs, then runs a dummy
inference to confirm the model is usable end-to-end.

Usage:
    conda run -n medasr_env python scripts/diagnose_fp16.py
"""
from pathlib import Path
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "medasr"
FP16_ONNX  = MODELS_DIR / "model_fp16.onnx"

def main():
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.")
        return

    print(f"ORT version : {ort.__version__}")
    print(f"Model       : {FP16_ONNX}")
    print()

    # Create session (CPU so we don't need GPU drivers)
    try:
        sess = ort.InferenceSession(str(FP16_ONNX), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"FAILED to create session: {e}")
        return

    print("=== Input metadata (as ORT sees it) ===")
    for inp in sess.get_inputs():
        print(f"  name={inp.name!r}  type={inp.type!r}  shape={inp.shape}")

    print()
    print("=== Output metadata ===")
    for out in sess.get_outputs():
        print(f"  name={out.name!r}  type={out.type!r}  shape={out.shape}")

    print()
    print("=== Dummy inference ===")
    # Use actual shapes: [1, T, 128].  Use T=100 frames for speed.
    T, N_MELS = 100, 128
    inp_meta = {inp.name: inp.type for inp in sess.get_inputs()}

    def make_input(name, shape, dtype_str):
        if dtype_str == "tensor(float16)":
            data = np.random.randn(*shape).astype(np.float16)
        elif dtype_str == "tensor(float)":
            data = np.random.randn(*shape).astype(np.float32)
        elif dtype_str in ("tensor(int32)", "tensor(int64)"):
            data = np.ones(shape, dtype=np.int32 if "32" in dtype_str else np.int64)
        else:
            data = np.random.randn(*shape).astype(np.float32)
        print(f"  {name}: numpy dtype={data.dtype}  shape={data.shape}")
        return data

    feeds = {
        "input_features": make_input("input_features", [1, T, N_MELS], inp_meta.get("input_features", "tensor(float)")),
        "attention_mask":  make_input("attention_mask",  [1, T],         inp_meta.get("attention_mask",  "tensor(int32)")),
    }

    try:
        outputs = sess.run(None, feeds)
        print(f"  Inference OK! logits shape={outputs[0].shape}  dtype={outputs[0].dtype}")
    except Exception as e:
        print(f"  Inference FAILED: {e}")

if __name__ == "__main__":
    main()
