# All-models shared-window benchmark

Date: 2026-08-30. Fixture: 18.714 s LibriVox window (same file for every family),
one warm-up then repeated measured runs; median reported. X-ASR excluded
(deprioritized streaming-latency model). Whisper leg is a single matrix pass;
other families run three measured repetitions.

| family | composition | median ms | median RTFx | oracle | status |
| --- | --- | ---: | ---: | --- | --- |
| whisper-large-v3-turbo | fp16io WebGPU encoder + GPU-KV greedy decoder | 682 | 27.45x | none | check |
| parakeet-tdt-v3 | fp16 WebGPU encoder + int8 WASM decoder (hybrid) | 923 | 20.44x | exact | pass |
| parakeet-tdt-v2 | fp16 WebGPU encoder + int8 WASM decoder (hybrid) | 510 | 37.15x | normalized | pass |
| gigaam-ctc | fp16 WebGPU encoder + CTC decode | 208 | 90.11x | none | pass |
| gigaam-rnnt | fp32 WebGPU encoder + WASM decoder/joint | 577 | 32.41x | none | pass |
| sensevoice-small | fp16 WebGPU + fp16 LUT CTC decode | 400 | 46.79x | none | pass |
| qwen3-asr-0.6b | fp32 WebGPU encoder + fp32 GPU-KV decoder | - | -x | none | error |

Manifest: tools/data/results/cross-model/all-models-shared-window-2026-08-30.json

## Notes

- Parakeet v3 int8 hybrid passed the exact 91-token oracle at 20.44x in this
  snapshot versus the 37.2x recorded on 2026-08-29, while v2 hit 37.15x in the
  same session - so the environment reaches the band and the v3 delta is
  either session variance beyond the documented 10-15% or a preprocessor
  difference to pin down (this run used the JS fbank; re-verify with the ONNX
  preprocessor before drawing conclusions).
- int8 decoder on the full-WebGPU composition produced garbled transcripts
  ("Prex rec public vol. or. b Wow") at ~5-8x during orchestrator bring-up.
  The INT8 decoder is transcript-safe only on WASM; keep it there (the
  library default already does).
- Qwen3-ASR 0.6B posted no result within the 15-minute runner budget on this
  window (no payload, exit 1); it completes only on the 11 s JFK fixture
  today. Consistent with the artifact-fragility pattern recorded in
  docs/GOAL_PROMPT.md (int4 hangs, fp16 hangs, graph capture rejected).
- Whisper 'check' status is the page's throughput-only label (oracle none),
  not a failure. Command history: this run used the fp16io-fp16 GPU-KV greedy
  composition (the 27x flagship path).

## Reproduction

Orchestrator: `webgpu-agent-test/scripts/run-all-models.mjs` (requires Vite
dev server on :8765). It sequences the per-family runners with the shared
window, parses each runner's payload, and writes this report plus the JSON
manifest. Whisper matrix case `en-greedy-gpu-kv-librivox` was added to
`scripts/run-webgpu-matrix.mjs` for this benchmark.
