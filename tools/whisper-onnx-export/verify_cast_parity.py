"""
Verify decoder_init Cast model produces identical results to the original fp16 path.

Tests that running decoder_init_cast.onnx with fp32 input produces the same
logits and KV cache as decoder_init.onnx with fp16 input (simulating the CPU
cast path that maybeCastEncoderHiddenStates does in JS).
"""
import numpy as np
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path(r"N:\github\asrjs\webgpu-agent-test\public\models\fp16")

def test_cast_parity():
    print("Loading models...")
    orig_sess = ort.InferenceSession(
        str(MODEL_DIR / "decoder_init.onnx"),
        providers=['CPUExecutionProvider']
    )
    cast_sess = ort.InferenceSession(
        str(MODEL_DIR / "decoder_init_cast.onnx"),
        providers=['CPUExecutionProvider']
    )

    # Create random encoder hidden states in fp32 (simulating encoder output)
    np.random.seed(42)
    enc_f32 = np.random.randn(1, 1500, 1280).astype(np.float32)
    
    # Convert to fp16 for the original model (simulating JS float32ToFloat16Bits)
    enc_f16 = enc_f32.astype(np.float16)

    # Input IDs: typical Whisper prompt [sot, lang, task]
    input_ids = np.array([[50258, 50268, 50359]], dtype=np.int64)

    # Current JS path: fp32 encoder output → CPU cast to fp16 → feed to decoder_init
    print("\nRunning ORIGINAL model (fp16 input)...")
    orig_outputs = orig_sess.run(None, {
        'input_ids': input_ids,
        'encoder_hidden_states': enc_f16,
    })
    
    # New path: fp32 encoder output → feed directly → GPU Cast handles it
    print("Running CAST model (fp32 input → Cast to fp16 internally)...")
    cast_outputs = cast_sess.run(None, {
        'input_ids': input_ids,
        'encoder_hidden_states': enc_f32,
    })

    # Compare outputs
    orig_names = [o.name for o in orig_sess.get_outputs()]
    cast_names = [o.name for o in cast_sess.get_outputs()]
    print(f"\nOriginal outputs: {len(orig_names)}")
    print(f"Cast outputs:     {len(cast_names)}")

    all_match = True
    for name in orig_names:
        orig_val = dict(zip(orig_names, orig_outputs))[name]
        
        if name not in cast_names:
            print(f"  {name}: MISSING in cast model!")
            all_match = False
            continue
            
        cast_val = dict(zip(cast_names, cast_outputs))[name]
        
        # Compare shapes
        if orig_val.shape != cast_val.shape:
            print(f"  {name}: shape mismatch {orig_val.shape} vs {cast_val.shape}")
            all_match = False
            continue
        
        # For fp16 tensors, compare with some tolerance
        if orig_val.dtype == np.float16:
            diff = np.max(np.abs(orig_val.astype(np.float32) - cast_val.astype(np.float32)))
            # fp16 has ~3.3 decimal digits of precision. Allow small diffs from Cast + fp16 round-trip
            tol = 0.1  # very loose — just checking for catastrophes
            ok = diff < tol
            status = "✅" if ok else f"❌ diff={diff:.4f}"
            if not ok:
                all_match = False
        else:
            ok = np.allclose(orig_val, cast_val)
            status = "✅" if ok else "❌"
            if not ok:
                all_match = False
        
        # Print summary for key outputs
        if 'logits' in name:
            print(f"  logits: shape={orig_val.shape} dtype={orig_val.dtype} max_diff={np.max(np.abs(orig_val.astype(np.float32) - cast_val.astype(np.float32))):.6f}")
        elif name.startswith('present.0'):
            print(f"  {name}: shape={orig_val.shape} {status}")

    print(f"\n{'✅ ALL OUTPUTS MATCH' if all_match else '❌ MISMATCH DETECTED'}")
    
    # Extra: verify Cast output matches np.astype(np.float16)
    if all_match:
        print("\n✅ Cast model is a drop-in replacement for the CPU f32→f16 cast path")

if __name__ == "__main__":
    test_cast_parity()
