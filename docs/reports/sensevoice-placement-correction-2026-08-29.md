# SenseVoice placement and stale-matrix correction (2026-08-29)

## Question

The early Chrome WebGPU matrix recorded SenseVoice small at 701 ms / RTFx
15.7. After GigaAM RNN-T's number proved stale, SenseVoice was re-measured
with ORT Web 1.29.0 on the 11.3 s jfk-short fixture (exact oracle), 1
warm-up + 3 measured runs, Chrome headless / NVIDIA Blackwell / D3D11.

## Results

| Composition | Median ms | RTFx | Prep ms | Encode ms | Decode ms | Parity |
|---|---|---|---|---|---|---|
| WebGPU (default) | 435.0 | 25.29x | 57.6 | 205.5 | 160.9 | exact |
| WASM, 8 threads | 1375.6 | 8.00x | 37.8 | 1945.5 (last run) | 119.7 | exact |

## Findings

1. The stale 15.7x is superseded: the current WebGPU default measures
   25.3x. No regression; the family is in good shape.
2. WebGPU placement is decisively correct for this single-graph encoder
   model: encode is roughly 10x faster than WASM (205.5 vs ~1945 ms).
   Unlike the one-frame-step decode loops (Parakeet TDT, GigaAM RNN-T),
   where WASM wins, SenseVoice validates the workload-specific placement
   rule in the opposite direction.
3. decodeMs (160.9 ms) is not an autoregressive loop - it is JS
   post-processing: full-vocabulary argmax over frameLength x vocab logits,
   CTC collapse with spans, tokenizer decode, and confidence/timing
   construction. This is the next phase-level target, best measured with
   the Node hot-path microbenchmark harness rather than noisy browser
   phases.
4. The phase-sum vs native-total gap is only ~10 ms, so the GPU->CPU
   logits readback is essentially attributed; no hidden transfer cost.
5. The model ships fp32-only (894 MB). An int8/fp16 export is the
   VRAM/size lever, requiring export tooling rather than placement.

## Harness changes

The SenseVoice browser runner previously hardcoded backend 'webgpu' and
cpuThreads 1. It now accepts backend and cpuThreads query parameters, and
the CLI exposes --backend=wasm and --cpu-threads=N, matching the other
family harnesses.

## Artifacts

tools/data/results/sensevoice/small-webgpu-jfk-3run.json,
tools/data/results/sensevoice/small-wasm-8t-jfk-3run.json

## Reproduction

node scripts/run-sensevoice-webgpu.mjs [--backend=wasm --cpu-threads=N] --warmup=1 --runs=3

## Cross-family placement picture after this slice

| Family | Structure | Fast path | Placement verdict |
|---|---|---|---|
| Parakeet TDT v3 | Conformer + GRU/TDT loop | ~36-37x | enc WebGPU, dec WASM |
| Parakeet TDT v2 | Conformer + GRU/TDT loop | ~55x | enc WebGPU, dec WASM |
| GigaAM RNN-T | Conformer + RNNT loop | ~25x | enc WebGPU, dec/joint WASM |
| SenseVoice small | Single encoder graph | ~25x | all WebGPU |
| GigaAM CTC | Single encoder graph | ~40x | WebGPU fp16 |

