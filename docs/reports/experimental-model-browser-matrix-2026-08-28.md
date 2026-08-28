# Experimental model browser matrix — 2026-08-28

This report records a fresh artifact-backed smoke pass against the current
`main` source at commit `ecf251e`. It validates runtime graph execution and
fixture transcript parity; it is not a representative WER study and does not
promote any family to a public preset.

## Environment and evidence boundary

- Host: Windows 10, NVIDIA Blackwell adapter.
- Browser harness: `N:\github\asrjs\webgpu-agent-test`, Vite on `:8765`,
  independent Chrome WebGPU launchers.
- Browser adapter: `navigator.gpu === true`, vendor `nvidia`, architecture
  `blackwell`.
- WASM harness: the repository's artifact-gated Vitest smoke tests using
  `onnxruntime-web`.
- Fixtures are implementation oracles. Transcript equality on one fixture does
  not establish human-label accuracy, language coverage, long-audio behavior,
  or production readiness.
- Large weights remain outside Git under `N:\models`.

## Current results

| Family / artifact | WASM result | Chrome WebGPU result | Chrome load / transcribe | Notes |
| --- | --- | --- | ---: | --- |
| GigaAM multilingual CTC, official fp32 and fp16 exports | Exact JFK transcript | Exact JFK transcript | 5.437 s / 0.408 s | fp16 WebGPU, RTFx 27.26; mixed-length batch also passes |
| GigaAM v3 E2E RNN-T, official `model.to_onnx` export | Exact official `example.wav` transcript | Exact official `example.wav` transcript | 8.476 s / 4.220 s | Russian-only fixture, RTFx 2.68 |
| SenseVoiceSmall, official FunASR ONNX export | Exact JFK transcript and `en` metadata | Exact JFK transcript and `en` metadata | 15.208 s / 2.240 s | RTFx 4.91; mixed-length batch also passes |
| X-ASR zh-en 160 ms, local Zipformer2 streaming graphs | Exact JFK transcript | Exact JFK transcript | 9.571 s / 38.007 s | Stateful encoder-cache path, RTFx is not reported by the harness |
| Qwen3-ASR 0.6B, official stacked graphs with dynamic encoder | Exact JFK transcript | Exact JFK transcript | 36.041 s / 5.940 s | Dynamic encoder, fp16 decoder, RTFx 1.85; 30-second model limit |

All browser results were `status: pass`, `webgpu: true`, and
`baselineMatch: true`. The Qwen browser run used
`audio-encoder-dynamic.onnx`; the WASM run used 1,050 input frames padded to
the model's 1,100-frame encoder shape and emitted 30 tokens.

Node-side WebGPU smoke remains classified as `WEBGPU_NO_ADAPTER` on this host.
That is an environment limitation and is kept separate from the successful
Chrome WebGPU evidence.

## Batch evidence

The two CTC families that expose `transcribeBatch()` were also tested with the
official graphs using the full JFK clip plus its first 60%. Both WASM and
independent Chrome WebGPU runs returned two results, preserved exact parity on
the full clip, and returned non-empty text for the shorter clip:

| Family | WASM batch result | Chrome WebGPU batch result |
| --- | --- | --- |
| GigaAM multilingual CTC fp32 | `batch_size=2`, first exact | `batchSize=2`, first exact, 0.724 s |
| SenseVoiceSmall | `batch_size=2`, first exact | `batchSize=2`, first exact, 1.088 s |

These measurements validate batch graph execution and output cardinality, not
batch scaling or broad quality. GigaAM RNN-T and Qwen remain offline single-clip
paths; X-ASR remains stateful encoder-cache streaming.

## Reproduction commands

From `N:\github\asrjs\speech-recognition`:

```powershell
$env:GIGAAM_CTC_ONNX_SMOKE = '1'
npm test -- --run tests/gigaam-ctc-onnx-backends.test.ts

$env:GIGAAM_RNNT_ONNX_SMOKE = '1'
npm test -- --run tests/gigaam-rnnt-onnx-backends.test.ts

$env:SENSEVOICE_ONNX_SMOKE = '1'
npm test -- --run tests/sensevoice-onnx-backends.test.ts

$env:XASR_ONNX_SMOKE = '1'
npm test -- --run tests/x-asr-onnx-backends.test.ts

$env:QWEN_OFFICIAL_ONNX_SMOKE = '1'
$env:NODE_OPTIONS = '--max-old-space-size=16384'
npm test -- --run tests/qwen3-asr-onnx-backends.test.ts
```

With the independent Vite server running from
`N:\github\asrjs\webgpu-agent-test`:

```powershell
node scripts/run-gigaam-webgpu.mjs
node scripts/run-gigaam-rnnt-webgpu.mjs
node scripts/run-sensevoice-webgpu.mjs
node scripts/run-gigaam-webgpu.mjs --batch
node scripts/run-sensevoice-webgpu.mjs --batch
node scripts/run-xasr-webgpu.mjs
node scripts/run-qwen-webgpu.mjs
```

The Node smoke tests write structured results to the local
`tools/data/results` directories. The Chrome launchers post structured results
to `N:\github\asrjs\webgpu-agent-test\_results`. These result files are
intentionally not model weights and remain local evidence.

## Promotion boundary

The five families remain artifact-gated experimental families. Before making a
public preset claim, add representative multilingual/quality coverage,
repeated-load and long-session memory measurements, and keep the exact
artifact/license provenance alongside the parity fixtures. Qwen remains
short-clip offline speech-LLM support; X-ASR is the only family in this matrix
with a true encoder-cache streaming contract.
