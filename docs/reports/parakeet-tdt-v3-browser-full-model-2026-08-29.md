# Parakeet TDT v3 browser full-model placement probe

Date: 2026-08-29
Workspace: `N:\github\asrjs\webgpu-agent-test`
Library checkout: `N:\github\asrjs\speech-recognition`

This report records the corrected full-model browser experiment that followed
the decoder-only GRU/WebGPU probe. It uses the existing Chrome headless
real-WebGPU harness, a real NVIDIA Blackwell adapter, the local
`nvidia/parakeet-tdt-0.6b-v3` artifacts, and the 18.714-second
`librivox.org.wav` fixture. The browser harness was run with the ORT Web 1.29
all-bundle so the ONNX `nemo128` preprocessor could execute its `Pad` node.

Representative passing commands (from `N:\github\asrjs\webgpu-agent-test`):

```powershell
node scripts/run-parakeet-tdt-webgpu.mjs --mode=wasm --encoder=fp16 `
  --preprocessor=onnx --decoder-quant=fp32 --audio-strategy=native
node scripts/run-parakeet-tdt-webgpu.mjs --mode=webgpu --encoder=fp16 `
  --preprocessor=js --decoder-quant=fp16 --audio-strategy=native
```

Raw browser captures remain in
`N:\github\asrjs\webgpu-agent-test\_results\`; the result payload records
the audio strategy, sample count, decoded duration, provider placement, and
transcript status.

The validation runner now defaults to `native-rate` for this model. Use
`--audio-strategy=target` only when intentionally measuring the browser
AudioContext resampler as a diagnostic control.

## Native reference controls

The same library path was run through Node WASM with the exact v3 artifacts.
All three controls emitted the complete 91-token reference transcript:

| Encoder | Decoder | Load | Transcribe | RTFx | Quality |
| ------- | ------- | ---: | ----------: | ---: | ------- |
| fp16 | fp16 | 5,500 ms | 17,997 ms | 1.04x | 91 tokens, exact |
| fp16 | fp32 | 5,482 ms | 18,384 ms | 1.02x | 91 tokens, exact |
| int8 | fp32 | 2,503 ms | 3,286 ms | 5.69x | 91 tokens, exact |

Machine-readable captures are stored beside this report:

- `parakeet-tdt-v3-fp16-fp16-node-2026-08-29.json`
- `parakeet-tdt-v3-fp16-fp32-node-2026-08-29.json`
- `parakeet-tdt-v3-int8-fp32-node-2026-08-29.json`

The previously accepted int8/int8 baseline remains in
`parakeet-tdt-v3-local-baseline-2026-08-26.md` at about 4.61x RTFx.

## Browser measurements

Each row is a single exploratory run unless otherwise noted. `status=fail`
means the run completed but did not match the 91-token expected transcript;
these are not promotion candidates.

| Encoder | Encoder EP | Preprocessor | Decoder EP/quant | Audio prep | Load | Transcribe | RTFx | Tokens | Status |
| ------- | ---------- | ------------ | --------------- | --------- | ---: | ----------: | ---: | -----: | ------ |
| fp16 | WebGPU | ONNX | WASM/fp32 | browser target-rate | 10,268 ms | 1,079 ms | 17.47x | 74 | fail |
| fp16 | WebGPU | ONNX | WASM/fp16 | browser target-rate | 9,847 ms | 1,164 ms | 16.18x | 74 | fail |
| fp16 | WebGPU | JS | WASM/fp32 | browser target-rate | 9,848 ms | 2,689 ms | 6.98x | 74 | fail |
| fp16 | WASM | ONNX | WASM/fp32 | browser target-rate | 11,926 ms | 23,069 ms | 0.81x | 74 | fail |
| fp16 | WASM | JS | WASM/fp32 | browser target-rate | 11,729 ms | 22,482 ms | 0.83x | 74 | fail |
| int8 | WebGPU | JS | WASM/fp32 | browser target-rate | 7,475 ms | 8,994 ms | 2.08x | 26 | fail |
| int8 | WASM | ONNX | WASM/int8 | browser target-rate | 6,998 ms | 7,552 ms | 2.48x | 24 | fail |
| fp16 | WebGPU | ONNX | WebGPU/fp32 | browser target-rate | 10,591 ms | 6,741 ms | 2.78x | 74 | fail |
| fp16 | WebGPU | ONNX | WASM/fp32 | native-rate linear | 10,314 ms | 1,010 ms | 18.66x | 91 | **pass** |
| fp16 | WebGPU | JS | WASM/fp32 | native-rate linear | 10,034 ms | 990 ms | 19.07x | 91 | **pass** |
| fp16 | WebGPU | JS | WASM/fp16 | native-rate linear | 9,481–9,557 ms | 1,083–1,167 ms | 16.14–17.41x | 91 | **pass** |
| fp16 | WebGPU | JS | WebGPU/fp16 | native-rate linear | 9,367–10,309 ms | 3,190–6,701 ms (median 3,324) | 5.64x median | 91 | **pass** |
| int8 | WebGPU | JS | WASM/int8 | native-rate linear | 7,658 ms | 9,294 ms | 2.02x | 91 | **pass** |

The fp32 encoder control could not be created in the browser because ORT Web
reported `Module.MountedFiles is not available` while resolving
`encoder-model.onnx.data`. The original narrow WebGPU bundle also rejected the
ONNX preprocessor's `Pad(13)` node; switching only the harness alias to the
1.29 all-bundle removed that provider error.

## Interpretation and next gate

The first seven browser rows used the default `browser-target-rate` AudioContext
resampler and stopped at 74 tokens despite the native controls being exact. A
follow-up run with the deterministic WAV parser and `native-rate` linear
resampling produced the complete 91-token transcript with the same fp16
encoder/WebGPU + ONNX-preprocessor + WASM-decoder composition. The audio was
18.714 s in both cases (the native path emitted 299,425 samples), so this is a
sample-value/preprocessing parity issue rather than truncation.

The native-rate rows establish two additional facts. First, ORT 1.29's full
WebGPU decoder is numerically correct on this workload, but its three-run
median (5.64x RTFx) is materially slower than the hybrid WebGPU-encoder/WASM-
decoder path (16–19x RTFx). Second, int8 is correct but slow on this WebGPU
encoder (2.02x RTFx), so quantization is a memory/size option rather than an
assumed throughput optimization.

These are validated placement candidates for this artifact and adapter, not a
blanket default change. Repeat them across browsers and artifacts, capture
warm-cache/memory behavior, and compare against the accepted int8/int8 Node
baseline before changing preset defaults. Keep the production composition
encoder-WebGPU/decoder-WASM, and make the model-specific audio strategy
explicit, until those broader gates pass.
