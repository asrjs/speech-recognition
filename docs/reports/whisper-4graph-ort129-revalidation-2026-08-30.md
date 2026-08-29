# Whisper 4-graph revalidation on ORT Web 1.29 (2026-08-30)

## Why

The Whisper Large V3 Turbo 2x to 26x case study was measured before the
onnxruntime-web 1.27-dev to 1.29.0 stable upgrade and the later family
work. This slice revalidates the flagship 4-graph splitgraph path on the
current stack: Chrome headless, NVIDIA Blackwell, D3D11, ORT Web 1.29.0,
30 s JFK clip, fp16io-fp16-webgpu local artifacts.

## Results

| Cell | RTFx | Total ms | Encode ms | Decode ms | Steps | KV |
|---|---|---|---|---|---|---|
| greedy + GPU-KV | 27.02x | 1106.6 | 183.2 | 707.1 | 49 | gpu-buffer |
| stable beam 2 | 5.57x | 5370.4 | 375.5 | 4764.3 | 98 | cpu |

Both produced the identical expected 50-token JFK transcript (greedy and
beam sequences match, as documented in the 2026-08-23 parity handoff).

## Comparison with recorded history

- Greedy GPU-KV: documented target was ~26x; the current stack measures
  27.0x. No regression from the ORT upgrade.
- Stable beam 2: recorded 10465.995 ms / 2.86x on 2026-08-23 and
  10914.095 ms / 2.74x on 2026-08-25; the current stack measures
  5370.4 ms / 5.57x - roughly 2x faster, reflecting the decoder
  optimizations landed since those handoffs.

## Verdict

The flagship Whisper path is validated on the current stack and faster
than any recorded beam-search baseline. The remaining recoverable warning
is whisper.decoder-align-legacy (decoder_align missing the causal-
self-attention export marker; timestamp interpolation fallback), which is
an artifact re-export item, not a runtime defect.

## Reproduction

node scripts/run-webgpu-matrix.mjs en-greedy-gpu-kv en-stable-beam-2

## Artifacts

tools/data/results/whisper/large-v3-turbo-4graph-greedy-gpukv-revalidation-2026-08-30.json,
tools/data/results/whisper/large-v3-turbo-4graph-stable-beam2-revalidation-2026-08-30.json

