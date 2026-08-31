#!/usr/bin/env python
"""Test different prompt_ids with the ONNX encoder."""
import numpy as np
import torch, torchaudio, onnxruntime as ort
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

mel = np.load("tools/data/results/nemotron/_tmp_mel.npy")

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
    return result["encoder_out"]


# Compute NeMo reference once
print("Loading NeMo for reference...")
nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu"
)
nemo.eval()
mel_t = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)
mel_len = torch.tensor([mel.shape[0]], dtype=torch.long)
with torch.no_grad():
    nemo_out, _ = nemo.encoder(audio_signal=mel_t, length=mel_len)
nemo_proj = nemo.joint.enc(nemo_out.transpose(1, 2))  # [1, 139, 640]
nemo_first4 = nemo_proj[0, :4, :]
print(f"NeMo projected first 4: max={float(nemo_first4.abs().max()):.3f}")

# Sweep prompt_ids
chunk = mel[:25][np.newaxis, ...].astype(np.float32)
print("\n=== prompt_id sweep ===")
for pid in [0, 1, 101, 100, 18, 64, 999, -1, 127]:
    first_caches = init_caches(enc_first)
    onnx_out = run_enc_onnx(enc_first, chunk, pid, first_caches, 0.0)
    onnx_first4 = torch.from_numpy(onnx_out[0])
    cos = float(torch.nn.functional.cosine_similarity(
        onnx_first4.flatten().unsqueeze(0),
        nemo_first4.flatten().unsqueeze(0)
    ))
    max_abs = float(onnx_first4.abs().max())
    diff_max = float((onnx_first4 - nemo_first4).abs().max())
    print(f"  pid={pid}: maxAbs={max_abs:.3f}, diffMax={diff_max:.3f}, cosSim={cos:.4f}")