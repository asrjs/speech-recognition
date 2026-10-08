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

## Re-export readiness

The old export blocker ("dynamo needs onnxscript, not installed") no longer
holds in this stack: transformers5 venv has torch 2.14.1+cpu with
onnxscript 0.7.2, so `torch.onnx.export` (dynamo) is available for the valid
fp32 streaming graphs — encoder (raw waveform, frame 5 ms / 80 samples,
in-graph Linear(80→320), causal convs, sliding-window attention, mask input)
and decoder (self+cross attention with cache). Invalid historical streaming
graphs from the 4.57 era stay out of production asset selection; keep their
failure records. Next slice: export, then parity against the oracle's token
ids, text and encoder-hidden hash above (fp16 after fp32 parity, per the
user's precision direction).
