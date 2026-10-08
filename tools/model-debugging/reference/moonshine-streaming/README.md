# Streaming Moonshine official oracle (project Transformers 5 stack)

Local snapshot `.cache/moonshine-streaming-tiny` is byte-identical to
`moonshine-ai/moonshine-streaming-tiny` at revision `f8e9dfd8c562…`
(size-verified per file; official org, not a community conversion). The old
`.cache/moonshine-streaming-remapped` layout and
`model.remapped.safetensors` are the pre-5.18 4.57-era layout (19/19 key names
differ) — do not use them with transformers 5; the original `model.safetensors`
matches `MoonshineStreamingForConditionalGeneration` exactly
(161 tensors, 0 missing / 0 unexpected / 0 shape mismatch).

## Oracle result (2026-10-08)

`official_oracle.py` (run with `tools/model-debugging/reference/transformers5`):

```powershell
tools/model-debugging/reference/transformers5/Scripts/python.exe tools/model-debugging/reference/moonshine-streaming/official_oracle.py --output tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-FRESH.json
```

On the pinned JFK effective PCM (`ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`):
`PASS_OFFICIAL_STREAMING_ORACLE` — 28 tokens, canonical transcript, decoder
cache stepwise==one-shot, fresh-instance reset equality, encoder hidden states
`[1, 550, 320]` captured with SHA-256 for later export parity. Receipt:
`tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-2026-10-08.json`.

## Contract finding: the attention mask is required

Generating without the processor's `attention_mask` degenerates into repeated
tokens on this whole-utterance path, while the offline moonshine model on the
same fixture/stack is mask-insensitive. The streaming encoder must receive the
padding mask; any faithful ONNX export or library path must carry mask
semantics (or bake all-ones behaviour with explicit length handling).

## Re-export progress (2026-10-08)

The old export blocker ("dynamo needs onnxscript, not installed") no longer
holds: transformers5 venv has torch 2.14.1+cpu with onnxscript 0.7.2.

**Encoder done — `PASS_STREAMING_ENCODER_FP32`**
(`export_streaming_encoder.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-encoder-fp32-2026-10-08.json`):
our own fp32 ONNX graph at `.cache/moonshine-streaming-onnx/encoder_model.onnx`
(opset 18, 377 nodes, dynamic `batch`/`time`, inputs `input_values` FLOAT +
`attention_mask` INT32 — the required mask from the oracle finding). The legacy
TorchScript tracer still fails (`invalid unordered_map<K,T> key`); the torch
dynamo exporter works. Parity: onnxruntime fp32 hidden vs torch fp32 hidden
max abs 4.26e-5 / mean 1.0e-6 / cosine 0.99999996, allclose atol/rtol 1e-4, and
end-to-end greedy tokens through the ONNX encoder equal the official oracle's
28 token ids and canonical text. Gate rationale recorded in the receipt: the
torch fp32 encoder is bit-stable in-process but not across processes (oneDNN
thread scheduling), so digests are identity evidence while the gate is explicit
fp32 tolerances plus exact token/text equality.

**Next**: the streaming decoder graph (self+cross attention with cache, the
prefill/step contract — reuse the offline cross-KV pattern from
`tools/moonshine-onnx-export/`), then fp16 after this fp32 gate, then the
chunked-state encoder refinement. Invalid historical streaming graphs stay out
of production asset selection; keep their failure records.
