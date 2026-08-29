# GigaAM RNN-T placement and threads matrix (2026-08-29)

## Question

The goal file recorded an early Chrome WebGPU matrix entry of 6,163 ms /
RTF 0.55 for GigaAM RNN-T, the slowest number in the cross-family matrix.
This slice re-measures the family with current artifacts (fp32-only
encoder 844 MB, decoder 4.4 MB, joint 2.6 MB), ORT Web 1.29.0, and the
per-component provider overrides, on the 11.29 s example.wav fixture
(exact Russian oracle), 1 warm-up + 3 measured runs, Chrome headless /
NVIDIA Blackwell / D3D11.

## Results

| Composition | Median ms | RTFx | Prep ms | Encode ms | Decode ms | Parity |
|---|---|---|---|---|---|---|
| enc WebGPU + dec/joint WASM (default), 1 thread | 449.3 | 25.13x | 130.8 | 68.4 | 236.0 | exact |
| enc WebGPU + dec/joint WASM, 8 threads | 451.8 | 24.99x | 117.1 | 73.8 | 184.6 | exact |
| all WebGPU (dec/joint on GPU) | 3889.4 | 2.90x | 75.1 | 64.9 | 3748.8 | exact |

## Findings

1. The current default composition is healthy: 25.1x RTFx with exact
   parity. The old 0.55x entry belongs to the all-WebGPU composition
   class, which still measures 2.9x today - a 16x decode-loop penalty
   (3748.8 ms vs 236.0 ms) from running the tiny per-token decoder/joint
   steps on GPU. The hybrid default is validated by a measured 8.7x
   end-to-end placement gap; this is the same one-frame-step lesson as
   Parakeet TDT.
2. Threads: 8 threads left the end-to-end median flat (451.8 vs 449.3 ms).
   The decode phase dropped 236.0 -> 184.6 ms but that sits inside the
   documented +-20% cross-session decode variance; no promotion claim.
3. Preprocessing is JS-only (GigaAmJsPreprocessor, no ONNX preprocessor
   artifact ships with the model). Prep measured 75-131 ms across sessions
   (noisy, 17-29% of total). A future ONNX/WebGPU fbank export could
   stabilize and speed this stage, mirroring the Parakeet nemo128 pattern.
4. The encoder is already fast on WebGPU (65-74 ms) and no quantized
   variants exist locally; an int8/fp16 encoder export is the remaining
   VRAM/size lever (844 MB fp32), requiring export tooling rather than a
   placement change.

## Harness changes

The GigaAM RNN-T browser runner previously hardcoded cpuThreads: 1. It now
accepts a cpuThreads query parameter, and the CLI runner exposes
--cpu-threads=N for thread diagnostics, matching the Parakeet harness.

## Artifacts

tools/data/results/gigaam/v3-rnnt-encgpu-decwasm-1t-librivox.json,
tools/data/results/gigaam/v3-rnnt-encgpu-decwasm-8t-librivox.json,
tools/data/results/gigaam/v3-rnnt-all-webgpu-librivox.json

## Reproduction

node scripts/run-gigaam-rnnt-webgpu.mjs [--cpu-threads=N] [--decoder-backend=webgpu --joint-backend=webgpu] --warmup=1 --repeat=3

