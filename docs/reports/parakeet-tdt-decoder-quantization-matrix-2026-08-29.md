# Parakeet TDT decoder quantization and v2/v3 matrix (2026-08-29)

## Question

The observed ~18-28x RTFx for Parakeet TDT in this library looked like a
regression against the 45-90x band remembered from the original parakeet.js
and transformers.js v4 work. Three hypotheses were tested on the same
18.714 s LibriVox fixture, same harness, same browser:

1. the per-step confidence softmax is a decode bottleneck;
2. INT8 decoder quantization transfers the historical parakeet.js
   throughput win to the current library;
3. the v3 GRU decoder graph itself is heavier than the v2 one.

## Setup

- Chrome headless/WebGPU, NVIDIA Blackwell, ORT Web 1.29.0
- fp16 encoder on WebGPU, ONNX preprocessor, WASM decoder (hybrid
  placement), deterministic native-rate linear WAV preparation
- 1 same-session warm-up, 5 measured runs, median reported
- audio: tools/data/fixtures/audio/librivox.org.wav (18.714 s)
- oracle: exact 91-token transcript (v3), normalized (v2)
- command: node scripts/run-parakeet-tdt-webgpu.mjs --mode=wasm
  --model={v2|v3} --encoder=fp16 --encoder-backend=webgpu
  --preprocessor=onnx --decoder-quant={fp32|int8} [--cpu-threads=8]
  [--confidence=off] --warmup=1 --repeat=5

## Results (saved artifacts in tools/data/results/nemo-tdt/)

| Config | Parity | Median ms | RTFx | Avg decode ms | Steps |
|---|---|---|---|---|---|
| v3 fp32 decoder, 12t | exact | 832.0 | 22.8x | 565.2 | 108 |
| v3 fp32 decoder, 12t, confidence off | exact | 719.3 | 26.2x | 463.7 | 108 |
| v3 int8 decoder, 8t | exact | 508.1 | 37.2x | 255.7 | 106 |
| v2 fp32 decoder, 8t | normalized | 511.2 | 36.9x | 254.0 | 123 |
| v2 int8 decoder, 8t | normalized | 541.5 | 34.9x | 279.3 | 123 |

Same-session pair for the confidence gate (earlier same-day runs):
742.8 ms with confidence vs 745.8 ms without - within noise.

## Findings

1. INT8 decoder is the dominant, reproducible v3 win. Decode phase
   dropped from ~565 ms to ~256 ms average (roughly halved), median total
   832 -> 508 ms, RTFx 22.8x -> 37.2x, with the exact 91-token transcript
   preserved. The library browser default (weights.ts decoderDefault
   'int8') is therefore validated as the fast exact path; the slow
   ~18-28x numbers came from parity probes that pinned the fp32 decoder.
2. The v3 decoder graph is genuinely heavier than v2 at fp32
   (~4.3-5.2 ms/step vs ~2.1-2.9 ms/step on comparable runs). The new GRU
   decoder suspicion is correct as a per-step cost factor, but INT8
   recovers most of it.
3. The confidence softmax gate is a measured no-op end-to-end (within
   session noise). Keep returnConfidence=false as a throughput option;
   do not cite it as a speedup. Earlier float32 logits-view borrowing had
   already made the softmax cheap.
4. v2 INT8 vs v2 fp32 flipped sign across sessions (466 vs 551 ms
   earlier; 541 vs 511 ms now). Treat INT8 on v2 as a size/memory option,
   not a guaranteed speed win - consistent with the goal rule that
   quantization is not automatically faster.
5. The historical 45-90x band is not a myth and not a library regression:
   v2 reproduces at ~35-41x on this 18.7 s clip in the same harness, and
   RTFx rises with clip length as fixed per-run costs amortize. The old
   demos' headline numbers came from v2 with fp32/int8/threaded-WASM
   configurations on longer clips.
6. Session-to-session variance on this host is ~10-15%; single-run
   comparisons below that threshold are noise.

## Actions

- Keep fp16 encoder (WebGPU) + int8 decoder (WASM) as the documented
  throughput-oriented browser composition for Parakeet TDT v3.
- The fp32 browser encoder control remains blocked by ORT Web
  external-data mounting (Module.MountedFiles is not available) for both
  v2 and v3; not pursued here.
- returnConfidence=false stays opt-in with no measured end-to-end claim.

