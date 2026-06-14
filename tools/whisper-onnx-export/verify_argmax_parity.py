"""
Verify ArgMax decoder_step produces correct next_token_id.

Compares the new next_token_id output against np.argmax(logits, axis=-1)
to confirm the graph surgery didn't break anything.
"""
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path

MODEL_DIR = Path(r"N:\github\asrjs\webgpu-agent-test\public\models\fp16")

def test_argmax_parity():
    print("Loading models...")
    orig_sess = ort.InferenceSession(
        str(MODEL_DIR / "decoder_step.onnx"),
        providers=['CPUExecutionProvider']
    )
    argmax_sess = ort.InferenceSession(
        str(MODEL_DIR / "decoder_step_argmax.onnx"),
        providers=['CPUExecutionProvider']
    )

    # Read manifest for model config
    with open(MODEL_DIR / "manifest.json") as f:
        manifest = json.load(f)

    config = manifest.get("config", {})
    num_layers = config.get("decoder_layers", 4)
    num_heads = config.get("decoder_attention_heads", 20)
    head_dim = config.get("d_model", 1280) // num_heads  # or from manifest

    # Try to get head_dim from manifest
    model_cfg = manifest.get("model_config", manifest.get("config", {}))
    d_model = model_cfg.get("d_model", 1280)
    head_dim = d_model // num_heads

    print(f"decoder_layers={num_layers}, heads={num_heads}, d_model={d_model}, head_dim={head_dim}")

    # Build feeds
    feeds = {}
    # input_ids: [1, 1] int64 — token 50360 (<|startoftranscript|>)
    feeds["input_ids"] = np.array([[50360]], dtype=np.int64)

    # Past KV for step (assuming first step: seq_len=1 for decoder, 1500 for encoder)
    for layer in range(num_layers):
        # Decoder KV: [1, 20, 1, 64] — first step has 1 past token
        feeds[f"past_key_values.{layer}.decoder.key"] = np.random.randn(1, num_heads, 1, head_dim).astype(np.float16)
        feeds[f"past_key_values.{layer}.decoder.value"] = np.random.randn(1, num_heads, 1, head_dim).astype(np.float16)
        # Encoder KV: [1, 20, 1500, 64]
        feeds[f"past_key_values.{layer}.encoder.key"] = np.random.randn(1, num_heads, 1500, head_dim).astype(np.float16)
        feeds[f"past_key_values.{layer}.encoder.value"] = np.random.randn(1, num_heads, 1500, head_dim).astype(np.float16)

    print(f"\nRunning inference with {len(feeds)} inputs...")

    # Run original model
    orig_outputs = orig_sess.run(None, feeds)
    orig_logits = orig_outputs[0]  # first output is logits
    print(f"Original logits shape: {orig_logits.shape}, dtype: {orig_logits.dtype}")

    # NumPy argmax
    np_argmax = np.argmax(orig_logits, axis=-1)  # [1, 1]
    print(f"NumPy argmax: {np_argmax}, value={np_argmax[0,0]}")

    # Run argmax model
    argmax_outputs = argmax_sess.run(None, feeds)
    # Find next_token_id and logits outputs
    output_names = [o.name for o in argmax_sess.get_outputs()]
    print(f"ArgMax model outputs: {output_names}")

    for name, val in zip(output_names, argmax_outputs):
        print(f"  {name}: shape={val.shape}, dtype={val.dtype}")

    # Find next_token_id
    next_token_id = None
    for name, val in zip(output_names, argmax_outputs):
        if name == "next_token_id":
            next_token_id = val
            break

    if next_token_id is None:
        print("ERROR: next_token_id not found in outputs!")
        return

    print(f"\nnext_token_id shape: {next_token_id.shape}, dtype: {next_token_id.dtype}")
    print(f"next_token_id value: {next_token_id}")

    # Verify parity
    np_next = np_argmax  # [1, 1] shape
    model_next = next_token_id  # should be [1, 1] INT32

    print(f"\nParity check:")
    print(f"  NumPy argmax:        {np_next} (shape={np_next.shape})")
    print(f"  Model next_token_id: {model_next} (shape={model_next.shape})")

    if np_next.shape == model_next.shape and np.all(np_next.astype(np.int32) == model_next):
        print("\n✅ PARITY VERIFIED — model ArgMax matches NumPy argmax")
    else:
        print(f"\n❌ MISMATCH!")
        if np_next.shape != model_next.shape:
            print(f"   Shape mismatch: {np_next.shape} vs {model_next.shape}")
        else:
            diff = np_next.astype(np.int32) - model_next
            print(f"   Difference: {diff}")

    # Also verify the logits output is unchanged from original
    argmax_logits = None
    for name, val in zip(output_names, argmax_outputs):
        if name == "logits":
            argmax_logits = val
            break

    if argmax_logits is not None:
        logits_match = np.allclose(orig_logits, argmax_logits, atol=0)
        print(f"\nLogits unchanged: {'✅ YES' if logits_match else '❌ NO'} (max diff: {np.max(np.abs(orig_logits.astype(np.float32) - argmax_logits.astype(np.float32)))})")

    # Verify KV outputs match
    kv_match = True
    for i in range(len(orig_outputs) - 1):  # skip logits
        if not np.allclose(orig_outputs[i+1], argmax_outputs[i+1]):  # +1 to skip logits/first
            kv_match = False
            break
    print(f"KV outputs unchanged: {'✅ YES' if kv_match else '⚠️  Check required'}")

if __name__ == "__main__":
    test_argmax_parity()
