# C84 — VAD smoke input identity

Status: bounded harness input-validation PASS, not model-backed VAD acceptance.

## Defect and fix

Both `tests/smoke/vad-integration-smoke.mjs` and `vad-pipeline-smoke.mjs`
assumed a 44-byte PCM16 WAV header. The existing JFK fixture contains RIFF
metadata: that code treated 17 extra shorts as audio, reading 176017 rather
than 176000 frames. It also accepted a copied fixture whose RIFF signature was
replaced with `FAKE`; downstream energy segmentation still printed PASS.

Both scripts now reuse the existing public `decodeWavToMonoPcm` export from
`dist/browser-media.js`, and print the actual float32 PCM SHA-256. There is no
new decoder, source-audio rewrite, resampler, core runtime or model-specific
implementation change. Existing segmentation criteria are preserved.

The four real subprocess regressions in `vad-audio-input.test.mjs` check each
script's exact frame count/PCM hash and reject non-RIFF input with nonzero exit
and no success banner. Invalid fixtures are copies in individually generated
temporary directories, cleaned in finally; canonical input is unchanged.
RUN_ASR is explicitly 0, so these tests cannot accidentally exercise or certify
the optional ASR branch. The initial four tests failed before these fixes;
the final four pass. Independent read-only review reports no findings.

## Current validation

Node v26.2.0; commands from repository root:

```powershell
node --test tests/smoke/vad-audio-input.test.mjs
node --check tests/smoke/vad-integration-smoke.mjs
node --check tests/smoke/vad-pipeline-smoke.mjs
node node_modules/vitest/vitest.mjs run tests/wav-decoder.test.ts tests/vad-controller.test.ts tests/realtime-vad-reset.test.ts
```

4 subprocess tests PASS; syntax checks PASS; 47 related Vitest tests PASS.
No full-suite, build, actual ASR or browser model run was needed for these
tool-only edits. The CLI checks exercise the existing build, not a newly built
distribution. Existing public WAV decoder has its separate 27 source tests.

Verified hashes:

- JFK WAV: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.
- 176000-frame f32le PCM: `ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
- Existing `dist/browser-media.js`: `83d9a3662a655cfccedda18c7637f7905aa6c7e87fdd8ee8d0569b7b595ed1d7`.
- Existing `dist/audio/wav.js`: `cec90d84e839a8955474457f24ff2474f6b9140152312b249e7778485399d0d8`.

## Remaining acceptance

The current TenVadBackend chunking adapter uses an energy/RMS fallback; these
smokes do not prove learned TEN-VAD or FireRed VAD inference, VAD accuracy,
physical microphone behavior, or injected asynchronous VAD reset on a model.
The existing adapter's introductory comments are not evidence of a worker
being instantiated. Optional ASR's nonempty/keyword checks also are not an
official token/text or numerical oracle and were not run here.

Real TEN assets exist in the sibling `vad-demo/public/wasm/ten_vad.js` and
`ten_vad.wasm`; their existence is a discovery lead, not a passed inference
gate or authorization for sibling writes. Next inspect the existing browser
adapter/worker and use these local assets for an actual model-backed VAD
fixture/reset test without replacing this input-identity regression.
