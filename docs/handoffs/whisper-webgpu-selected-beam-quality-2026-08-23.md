# Whisper WebGPU Selected-Beam Quality Handover

Date: 2026-08-24
Branch: `feat/whisper-cleanup-beam-temperature`
Workspace: `N:\github\asrjs\speech-recognition`
Previous checkpoint: `ec24cef feat(whisper): finish beam quality and alignment pipeline`

## What Landed

Selected-beam quality metrics no longer require retaining full-vocabulary
logits for every beam.

- Beam search records scalar `{tokenId, logProb, entropy}` traces on each
  hypothesis and returns traces only for the **winning sequence**.
- Greedy/sampling collect the same traces only when `trackQuality: true`, so
  the WebGPU greedy fast path does not pay extra entropy work by default.
- Logprob and entropy gates prefer those traces over `Float32Array` vocab
  tensors. The raw decoder-init vector is still copied once for no-speech.
- `EnhancedWhisperExecutor` sets `trackQuality: true`, does not snapshot
  per-token vocab logits, and still forwards caller `onTokenLogits`.
- GPU-KV greedy computes traces only when `trackQuality` is set.
- Merged-decoder greedy/beam attach selected-sequence traces.

Canonical transcript mapping is unchanged. Traces live on native decode
results (`WhisperDecodeResult.tokenTraces` /
`WhisperNativeTranscript.tokenTraces`).

## Model Variants (unchanged)

WebGPU speed-path target:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
encoder: fp16_iofp32_fp16out   (remote preset names fp16_iofp32/encoder_model.onnx)
decoder: fp16
preset:  fp16io-fp16-webgpu
```

Do not substitute `onnx-community/whisper-large-v3-turbo` for speed-path
validation. Healthy FP16 evidence remains the independent `25.6993x` JFK run
from the previous handover. The follow-up browser matrix is recorded in the
[browser parity handover](whisper-webgpu-browser-parity-2026-08-23.md): stable
and batched CPU-KV beam matched exactly for English beam 5, timestamped English
beam 2, and Turkish auto beam 2.

## Verification

From `N:\github\asrjs\speech-recognition`:

```powershell
npm test -- --run
npm run typecheck
npm run lint
npm run build
```

Results:

- Tests: 115 files passed, 1 skipped; 720 tests passed, 4 skipped
- Typecheck passed
- Build passed
- Lint: 0 errors, 6 existing warnings

Focused coverage for this slice:

```powershell
npx vitest run tests/quality-gates.test.ts tests/whisper-enhanced-executor.test.ts tests/whisper-beam-search-decode.test.ts tests/whisper-core-score.test.ts tests/inference-math.test.ts --reporter=dot
```

## Next Implementation Order

1. Revalidate browser parity on the custom splitgraph model for English and a
   Turkish fixture. Keep stable CPU-KV beam as the correctness oracle.
2. Keep `experimentalBatchedBeam` opt-in until broader beam-size, EOS,
   timestamp, and Turkish parity coverage is complete.
3. Keep GPU-KV restricted to greedy argmax. Do not enable GPU-KV beam without
   correct KV cloning/reordering and output parity.

`condition_on_previous_text`, hotwords, and numeral suppression remain
deprioritized.
