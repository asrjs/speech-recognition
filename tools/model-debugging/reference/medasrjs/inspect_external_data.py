"""
Inspect the external data filename(s) embedded inside an ONNX model.
Reads only the graph/proto structure - does NOT load tensor data.
Usage: python scripts/inspect_external_data.py
"""
from pathlib import Path
import onnx

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "medasr"

def inspect(onnx_path: Path):
    print(f"\nInspecting: {onnx_path}")
    if not onnx_path.exists():
        print("  [MISSING]")
        return
    m = onnx.load(str(onnx_path), load_external_data=False)
    seen = {}
    for t in m.graph.initializer:
        if t.data_location == 1:  # EXTERNAL
            for loc in t.external_data:
                if loc.key == "location":
                    seen[loc.value] = seen.get(loc.value, 0) + 1
    if seen:
        for path, count in seen.items():
            print(f"  external data location: {repr(path)}  ({count} tensors)")
    else:
        print("  No external data (all weights inline)")

    # Also show IO types
    ELEM_NAMES = {1: "float32", 10: "float16", 6: "int32", 7: "int64"}
    print("  Inputs:")
    for inp in m.graph.input:
        t = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: {ELEM_NAMES.get(t, f'type={t}')}")
    print("  Outputs:")
    for out in m.graph.output:
        t = out.type.tensor_type.elem_type
        print(f"    {out.name}: {ELEM_NAMES.get(t, f'type={t}')}")

if __name__ == "__main__":
    inspect(MODELS_DIR / "model.onnx")
    inspect(MODELS_DIR / "model_fp16.onnx")
    inspect(MODELS_DIR / "model_int8.onnx")
