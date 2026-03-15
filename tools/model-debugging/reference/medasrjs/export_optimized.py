"""
Create optimized variants of the FP32 MedASR ONNX model (FP16 and INT8 dynamically quantized).
"""

import argparse
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    from onnxconverter_common import float16
except ImportError:
    print("Please install onnxconverter_common: pip install onnxconverter_common")
    exit(1)

def optimize_models(onnx_model_path: Path):
    if not onnx_model_path.exists():
        print(f"Error: Base FP32 model not found at {onnx_model_path}")
        return

    print(f"Loading Base FP32 Model: {onnx_model_path}")
    model = onnx.load(str(onnx_model_path))

    # --- FP16 Conversion ---
    print("\n[FP16 Conversion]")
    fp16_model_path = onnx_model_path.parent / "model_fp16.onnx"
    try:
        from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision
        import numpy as np
        
        # MedASR takes input_features: float32, attention_mask: int64 or int32
        # Provide deterministic inputs to allow precision validation
        np.random.seed(42)
        dummy_feats = (np.random.randn(1, 100, 128) * 0.1).astype(np.float32)
        dummy_mask = np.ones((1, 100), dtype=np.int32)
        # Verify graph type for attention_mask
        for inp in model.graph.input:
            if inp.name == "attention_mask":
                if inp.type.tensor_type.elem_type == onnx.TensorProto.INT64:
                    dummy_mask = dummy_mask.astype(np.int64)
                    
        feed_dict = {
            "input_features": dummy_feats,
            "attention_mask": dummy_mask
        }
        
        model_fp16 = auto_convert_mixed_precision(
            model,
            feed_dict,
            rtol=0.1,
            atol=0.1,
            keep_io_types=True
        )
        onnx.save(model_fp16, str(fp16_model_path))
        print(f"Successfully saved FP16 (mixed precision) model to {fp16_model_path}")
    except Exception as e:
        print(f"Failed to create FP16 model: {e}")

    # --- INT8 Dynamic Quantization ---
    print("\n[INT8 Dynamic Quantization]")
    int8_model_path = onnx_model_path.parent / "model_int8.onnx"
    try:
        # We use dynamic quantization because it doesn't require a calibration dataset, 
        # which is usually fine for encoder-only models.
        quantize_dynamic(
            model_input=str(onnx_model_path),
            model_output=str(int8_model_path),
            weight_type=QuantType.QUInt8 # QInt8 or QUInt8
        )
        print(f"Successfully saved INT8 model to {int8_model_path}")
    except Exception as e:
        print(f"Failed to create INT8 model: {e}")
        
    # Sizes
    for p in [onnx_model_path, fp16_model_path, int8_model_path]:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"[{p.name}] Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_model_path = Path(__file__).resolve().parent.parent / "models" / "medasr" / "model.onnx"
    parser.add_argument("--onnx-path", type=Path, default=default_model_path)
    args = parser.parse_args()
    optimize_models(args.onnx_path)
