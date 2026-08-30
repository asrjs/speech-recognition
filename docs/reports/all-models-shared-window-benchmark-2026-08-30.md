# All-models shared-window benchmark

Date: 2026-08-30. Fixture: 18.714 s LibriVox window (same file for every family),
one warm-up then repeated measured runs; median reported. X-ASR excluded
(deprioritized streaming-latency model). Whisper leg is a single matrix pass;
other families run three measured repetitions.

| family | composition | median ms | median RTFx | oracle | status |
| --- | --- | ---: | ---: | --- | --- |
| whisper-large-v3-turbo | fp16io WebGPU encoder + GPU-KV greedy decoder | 681 | 27.49x | none | check |
| parakeet-tdt-v3 | fp16 WebGPU encoder + int8 WASM decoder (hybrid) | 843 | 22.33x | exact | pass |
| parakeet-tdt-v2 | fp16 WebGPU encoder + int8 WASM decoder (hybrid) | 613 | 30.75x | normalized | pass |
| gigaam-ctc | fp16 WebGPU encoder + CTC decode | 190 | 98.36x | none | pass |
| gigaam-rnnt | fp32 WebGPU encoder + WASM decoder/joint | 579 | 32.30x | none | pass |
| sensevoice-small | fp16 WebGPU + fp16 LUT CTC decode | 381 | 49.07x | none | pass |
| qwen3-asr-0.6b | fp32 WebGPU encoder + fp32 GPU-KV decoder | 6996 | 2.65x | none | pass |

Manifest: tools/data/results/cross-model/all-models-shared-window-2026-08-30.json

## Same-day repeat run (page-fix validation + variance sample)

A second full run of the orchestrator on the same day, after fixing the
harness page defect and with the repaired page filling the Qwen row:

| family | run 1 RTFx | run 2 RTFx | delta |
| --- | ---: | ---: | --- |
| whisper-large-v3-turbo | 27.45x | 27.49x | +-0.1% |
| parakeet-tdt-v3 | 20.44x | 22.33x | +9% |
| parakeet-tdt-v2 | 37.15x | 30.75x | -17% |
| gigaam-ctc | 90.11x | 98.36x | +9% |
| gigaam-rnnt | 32.41x | 32.30x | -0.3% |
| sensevoice-small | 46.79x | 49.07x | +5% |
| qwen3-asr-0.6b | error | 2.65x (6996 ms) | first valid sample |

Takeaways:

- Whisper GPU-KV greedy is extremely stable across sessions on this window
  (and matches the 27.02x 30 s revalidation), making it the best reference
  leg for detecting machine-state drift.
- The WASM-decode-loop families (Parakeet v2/v3) swing the most between
  sessions (up to ~20%); treat their single-session numbers as bands, not
  points, and prefer paired same-session A/Bs for promotion decisions.
- Qwen now completes on the shared 18.7 s window at 2.65x, consistent with
  its 2.08-2.19x on the 11 s JFK fixture within the same environment state;
  the earlier "no payload" row was the harness page defect, not a window-
  length or artifact limitation.
