#!/usr/bin/env python
"""Compare NeMo joint argmax vs ONNX joiner argmax using NeMo encoder features."""
import numpy as np
import torch, onnxruntime as ort
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

print("Loading NeMo + ONNX...")
nemo = EncDecRNNTBPEModelWithPrompt.restore_from(
    "N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo",
    map_location="cpu"
)
nemo.eval()
nemo.set_inference_prompt("en")
decoder_sess = ort.InferenceSession(
    "N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/decoder.onnx",
    providers=["CPUExecutionProvider"]
)
joiner_sess = ort.InferenceSession(
    "N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/joiner.onnx",
    providers=["CPUExecutionProvider"]
)
vocab = {}
for line in open("N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx/tokens.txt", encoding="utf-8"):
    parts = line.split("\t", 1)
    if len(parts) == 2:
        vocab[int(parts[0])] = parts[1]

mel = np.load("tools/data/results/nemotron/_tmp_mel.npy")
mel_t = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)
mel_len = torch.tensor([mel.shape[0]], dtype=torch.long)
with torch.no_grad():
    nemo_enc, _ = nemo.encoder(audio_signal=mel_t, length=mel_len)
    enc_projected = nemo.joint.enc(nemo_enc.transpose(1, 2))  # [1, 139, 640]
enc_np = enc_projected.squeeze(0).numpy()  # [139, 640]

# Initial decoder with blank token
h = torch.zeros(2, 1, 640)
c = torch.zeros(2, 1, 640)
g_raw, hid = nemo.decoder.predict(y=torch.tensor([[13087]]), state=(h, c), add_sos=False)
h, c = hid
g_proj = nemo.joint.pred(g_raw)  # apply joint.pred (projected g, matching ONNX decoder_out)
g_proj = g_proj.squeeze(0)  # [1, 640]
print("\n=== Frame 0: NeMo raw g (via pred) vs ONNX decoder_out ===")
neMo_logits = nemo.joint.joint_after_projection(f=enc_projected[:, 0:1, :], g=g_proj.unsqueeze(0))  # [1, 1, 640]

# ONNX decoder produces projected g
onnx_g, onnx_h, onnx_c = decoder_sess.run(None, {
    "token": np.array([[13087]], dtype=np.int64),
    "h_in": np.zeros([2, 1, 640], dtype=np.float32),
    "c_in": np.zeros([2, 1, 640], dtype=np.float32),
})
print(f"ONNX decoder_out shape: {onnx_g.shape}, maxAbs: {float(np.abs(onnx_g).max()):.3f}")
onnx_logits = joiner_sess.run(None, {
    "encoder_frame": enc_np[0:1, :].astype(np.float32),
    "decoder_out": onnx_g.reshape(1, 640),
})[0]
print(f"ONNX logits argmax: {int(np.argmax(onnx_logits))}, top-5: {np.argsort(onnx_logits[0])[-5:].tolist()}")
print(f"NeMo vs ONNX logits maxDiff: {float((neMo_logits.detach().numpy() - onnx_logits).max()):.4f}")