#!/usr/bin/env python
"""Compare NeMo encoder vs ONNX encoder outputs."""
import numpy as np
import torch, torchaudio, onnxruntime as ort
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

mel = np.load("tools/data/results/nemotron/_tmp_mel.npy")

# Load ONNX encoder
enc_first = ort.InferenceSession(
    "N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/encoder_320ms_first_fp16.onnx",
    providers=["CPUExecutionProvider"]
)


def init_caches(sess):
    caches = {}
    for inp in sess.get_inputs():
        if inp.name in ("input_features", "prompt_ids", "cache_mask"):
            continue
        shape = [max(1, d) for d in inp.shape]
        dtype = np.int64 if "int64" in inp.type else np.float32
        caches[inp.name] = np.zeros(shape, dtype=dtype)
    return caches


def run_enc_onnx(sess, chunk, prompt_id, caches, cache_mask_value):
    feeds = {
        "input_features": chunk.astype(np.float32),
        "prompt_ids": np.array([prompt_id], dtype=np.int64),
        "cache_mask": np.full([1, 1, 1, 60], cache_mask_value, dtype=np.float32),
    }
    feeds.update(caches)
    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]
    result = dict(zip(out_names, outs))
    new_caches = {}
    for k, v in result.items():
        if k == "encoder_out":
            continue
        if "_out_" in k:
            base, _, tail = k.partition("_out_")
            new_caches[f"{base}_{tail}"] = v
        else:
            new_caches[k] = v
    return result["encoder_out"], new_caches


# Run ONNX first encoder with first 25 mel frames
first_caches = init_caches(enc_first)
chunk = mel[:25][np.newaxis, ...].astype(np.float32)
onnx_out, _ = run_enc_onnx(enc_first, chunk, 0, first_caches, 0.0)
print(f"ONNX enc1 shape: {onnx_out.shape}, maxAbs={float(np.abs(onnx_out).max()):.3f}, meanAbs={float(np.abs(onnx_out).mean()):.4f}")

# Run NeMo encoder on same mel
print("\nLoading NeMo model...")
nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu"
)
nemo.eval()
nemo.set_inference_prompt("en")

# Get the encoder
encoder = nemo.encoder
mel_t = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)  # [1, 128, T]
mel_len = torch.tensor([mel.shape[0]], dtype=torch.long)
prompt = torch.tensor([0], dtype=torch.long).unsqueeze(0)  # [1, 1]? or [1]?
print(f"mel_t shape: {mel_t.shape}")

# Try calling encoder (no prompt_ids at encoder level; handled at model level)
with torch.no_grad():
    # Initialize caches to zeros
    cache_last_channel = None
    cache_last_time = None
    nemo_out, nemo_out_len = encoder(audio_signal=mel_t, length=mel_len)
# nemo_out is [B, D, T] or [B, T, D]?
print(f"NeMo out shape: {nemo_out.shape}, dtype: {nemo_out.dtype}")
print(f"NeMo out stats: max={float(nemo_out.max()):.3f}, mean={float(nemo_out.mean()):.4f}")

# Apply joint.enc projection to nemo_out to match ONNX encoder_out dim (640)
joint = nemo.joint
enc_proj = joint.enc  # Linear(1024, 640)
nemo_projected = enc_proj(nemo_out.transpose(1, 2))  # [B, T, 640]
print(f"NeMo projected shape: {nemo_projected.shape}, max={float(nemo_projected.max()):.3f}, mean={float(nemo_projected.mean()):.4f}")

# Compare first 4 frames (matching ONNX enc output)
onnx_first4 = torch.from_numpy(onnx_out[0])  # [4, 640]
nemo_first4 = nemo_projected[0, :4, :]  # [4, 640]
diff = (onnx_first4 - nemo_first4).abs()
print(f"\nComparing first 4 encoder frames:")
print(f"  ONNX: max={float(onnx_first4.abs().max()):.3f}, mean={float(onnx_first4.abs().mean()):.4f}")
print(f"  NeMo: max={float(nemo_first4.abs().max()):.3f}, mean={float(nemo_first4.abs().mean()):.4f}")
print(f"  Diff: max={float(diff.max()):.3f}, mean={float(diff.mean()):.4f}")
print(f"  Cosine sim: {float(torch.nn.functional.cosine_similarity(onnx_first4.flatten().unsqueeze(0), nemo_first4.flatten().unsqueeze(0))):.4f}")