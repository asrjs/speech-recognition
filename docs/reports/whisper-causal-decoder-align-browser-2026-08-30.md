# Whisper causal decoder_align browser validation (2026-08-30)

## Why

The last recoverable warning on the flagship Whisper path was
`whisper.decoder-align-legacy`: the published browser preset
(`ysdede/whisper-large-v3-turbo-onnx-4graph`, fp16 folder) ships a
`decoder_align.onnx` export that lacks the `alignment_export.causal_self_attention`
marker, so the executor falls back to post-hoc timestamp interpolation and the
runtime emits a warning on every timestamped run.

A locally validated causal re-export family exists at
`N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5` (Node validation:
`N:\models\_asrjs-whisper-r5-validation-20260825\report-with-baseline.md`).
Its manifest carries:

- `causal_self_attention: true`
- `encoder_hidden_state_dtype: float16`
- `attention_values: logits`, `attention_layout: selected_heads`

The 2026-08-25 handoff proved the pairing in a *temporary* directory. This
slice makes it a first-class, persistent route in the browser harness and
captures committed evidence.

## What changed (harness, local only - not a git repo)

- `webgpu-agent-test/gigaam-asset-plugin.mjs`: new asset route
  `/models/fp16-causal/` serving the r5 folder.
  Middleware prefix intercepts before the static `public/models/fp16` folder.
- `webgpu-agent-test/src/main.js`: two new presets:
  - `fp16io-causal-webgpu` = proven historical pairing (public fp16io
    encoder folder + r5 causal decoder family).
  - `causal-r5-full-webgpu` = self-consistent r5 family (all four graphs).
- `webgpu-agent-test/scripts/run-webgpu-matrix.mjs`: five new cases
  (`causal-*`).

Library code in `speech-recognition` is unchanged: the
`attention_values: logits` / `selected_heads` fast-alignment path and the
manifest gate were already implemented; this slice only proves them against
the causal artifact in the real Chrome/WebGPU harness.

## Results (headless Chrome, ORT Web 1.29, NVIDIA Blackwell D3D11)

All cells: exact expected transcript, `wordAlignmentSource: "fast"`,
`decoderGpuTensorDownloads: 0`, and **empty warnings array** - the
`whisper.decoder-align-legacy` warning is gone on the causal artifact.

| Case | Preset | Audio | RTFx | total ms | first word |
| --- | --- | --- | --- | --- | --- |
| en-greedy-timestamps | fp16io-causal | JFK 10 s | 13.75 | 720.7 | In 2.12-2.84 s |
| stable beam 2 | fp16io-causal | JFK 10 s | 3.97 | 2520.3 | In 2.12-2.84 s |
| batched beam 2 | fp16io-causal | JFK 10 s | 5.02 | 1985.8 | In 2.12-2.84 s |
| greedy + GPU-KV | causal-r5-full | JFK 10 s | 13.51 | 735.2 | In 2.12-2.84 s |
| greedy + GPU-KV | fp16io-causal | JFK 30 s | **27.02** | 1106.8 | In 2.04-2.84 s |

- First-word anchor matches the 2026-08-25 temporary-dir proof (2.12-2.84 s;
  faster-whisper reference 2.16-2.84 s) and the documented fp16-encoder frame
  quantization boundary.
- The 30 s greedy cell measures 27.0197x - **identical** to the recorded
  no-timestamps revalidation (27.02x). Enabling timestamps on the causal
  align graph costs nothing measurable in decode (44 steps, 610.5 ms, GPU-KV).
- Both preset variants (mixed public-encoder pairing and fully self-consistent
  r5 family) pass; transcripts and anchors are identical.

Evidence JSONs: `tools/data/results/whisper/large-v3-turbo-4graph-causal-*-2026-08-30.json`
(warm-up + measurement pairs; measurement cells reported above).

## Verdict

The runtime-side decoder-align frontier is **closed**: with a
marker-carrying artifact the executor takes the fast causal path, produces
faster-whisper-comparable word anchors, warns nothing, and keeps full greedy
throughput. `causal-r5-full-webgpu` is additionally viable as the single-folder
option (one coherent export family, same numbers).

## Remaining boundary (artifact publishing - outside this slice)

The **public HF preset still ships the legacy align graph**, so default
downloads keep the recoverable fallback warning until a re-export is
published. Publishing requires explicit user approval per project policy:
re-export all four precision variants with the causal flags, run
`audit_publish.py`/`test_kv_export.py` gates, and upload to
`ysdede/whisper-large-v3-turbo-onnx-4graph`. Until then, local users can
point the decoder folder at a causal family (as the two new harness presets do).

## Repro

```powershell
cd N:/github/asrjs/webgpu-agent-test
npm run dev   # :8765
node scripts/run-webgpu-matrix.mjs causal-en-greedy-timestamps causal-en-stable-beam-timestamps causal-en-batched-beam-timestamps causal-r5full-en-greedy-timestamps causal-en-greedy-timestamps-30s
```
