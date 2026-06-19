# Whisper WebGPU Completion Validation Handoff

Date: 2026-06-19
Branch: `feat/whisper-cleanup-beam-temperature`

## What Changed

- Added the current implementation plan:
  `docs/plans/whisper-webgpu-completion-plan.md`
- Fixed enhanced temperature fallback so each retry temperature is passed to the
  vanilla Whisper executor in both single-chunk and VAD chunk paths.
- Preserved caller-provided `onTokenLogits` callbacks while the enhanced wrapper
  collects logits for quality gates.
- Fixed `withTemperatureFallback().attempts` so it reports decode attempts, not
  the number of quality-gate evaluations.
- Extracted Whisper language-token selection into
  `src/models/whisper-seq2seq/language-detection.ts`.
- Wired splitgraph language probing to use that helper.
- Fixed core beam-search survivor KV-cache alignment when a completed beam is
  retained while active beams continue due to `patience`.
- Fixed fp16 splitgraph beam KV bridging in browser/WebGPU:
  - raw fp16 KV arrays are reconstructed as `Float16Array` inputs when ORT
    requires that constructor;
  - per-beam KV tensor dims are carried with each cache entry instead of using
    only a global dims table.
- Tightened decode dispatch to match Whisper/faster-whisper: `temperature=0`
  uses greedy/beam argmax, nonzero temperature uses sampling, and `bestOf` only
  applies to nonzero-temperature sampling.
- Added opt-in experimental batched beam decode for the CPU-KV splitgraph path:
  active beams can share one `decoder_step` ORT call when
  `experimentalBatchedBeam` is true and the model accepts batch-shaped step
  inputs.

## Local Verification Already Run

From `N:\github\asrjs\speech-recognition`:

```powershell
npm test -- --run
npm run typecheck
npm run lint
npm run build
```

Final local result:

```text
Vitest: 112 files passed, 1 skipped; 653 tests passed, 4 skipped
Typecheck: passed
Lint: 0 errors, 6 existing warnings
Build: passed
```

Focused suites used while developing:

```powershell
npm test -- --run tests/whisper-fp16-kv-input.test.ts tests/whisper-beam-search-decode.test.ts tests/whisper-core-score.test.ts tests/whisper-splitgraph-decode.test.ts
npm run typecheck
```

Earlier wrapper/language suite also passed:

```powershell
npm test -- --run tests/whisper-language-detection.test.ts tests/whisper-enhanced-executor.test.ts tests/quality-gates.test.ts tests/whisper-temperature-fallback.test.ts tests/whisper-beam-search-decode.test.ts
```

## Browser Harness Report

Harness:

```text
N:\github\asrjs\webgpu-agent-test
```

The tester linked `node_modules\@asrjs\speech-recognition` to this working tree,
used Chrome `149.0.7827.155` headless, confirmed `navigator.gpu=true`,
`crossOriginIsolated=true`, and intercepted result posts so no new `_results`
files were written.

Greedy GPU-KV sanity:

```text
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1
```

Result: functional pass. Transcript prefix begins:

```text
In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I
```

Representative metrics: total about 1069ms, decode about 783ms, RTFx about
28.29, decoder step p50 about 11.6ms, p95 about 15.4ms, GPU tensor downloads 0,
KV location `gpu-buffer`.

Stable beam validation:

```text
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&numBeams=2&patience=1
```

Result: functional pass. Transcript prefix begins:

```text
In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of maximum danger. I do not shrink from this responsibility. I welcome it. I do not believe that any of us would exchange
```

Paired measurement metrics from the batched-beam validation run: total
`15156.185ms`, transcribe `15155.660ms`, encode `471.455ms`, decode
`14538.675ms`, RTFx `1.9755`, decoder init `105.390ms`, decoder step
`12392.930ms`, step p50 `126.210ms`, step p95 `140.470ms`, step count `98`,
KV location `cpu`.

Experimental batched beam validation:

```text
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&numBeams=2&patience=1&batchedBeam=1
```

Result: functional pass with the same transcript prefix as stable beam.
Metadata confirmed `decoding.experimentalBatchedBeam: true`; no ONNX/ORT
batch-shaped decoder input rejection occurred.

Paired measurement metrics: total `12834.765ms`, transcribe `12834.330ms`,
encode `485.435ms`, decode `12247.235ms`, RTFx `2.3320`, decoder init
`95.250ms`, decoder step `10345.335ms`, step p50 `212.005ms`, step p95
`218.835ms`, step count `49`, KV location `cpu`.

Interpretation: the batched path halves decoder-step ORT calls for two active
beams and improved this paired browser measurement by about 15%. Per-call time is
higher because each ORT run carries a beam-shaped batch.

## Browser Harness To Test

Fast test harness:

```text
N:\github\asrjs\webgpu-agent-test
```

Recommended validation order:

1. Sync or point the harness at the local `speech-recognition` working tree.
2. Run the default greedy WebGPU path with `experimentalGpuKvCache=true`.
3. Run the stable beam path with GPU-KV disabled.
4. Run the experimental batched beam path by adding `batchedBeam=1`.
5. Run a temperature-fallback scenario with GPU-KV disabled, since GPU-KV is
   intentionally greedy/temperature-0 only.

## Expected Behavior

- Greedy WebGPU GPU-KV remains the fast path and should still produce the known
  JFK transcript prefix without token-policy regressions.
- Beam search should run on the splitgraph CPU-KV path. It should not use
  `experimentalGpuKvCache`.
- `experimentalBatchedBeam` should only change how active beam steps are grouped
  into ORT calls. Token prefix, EOS behavior, timestamp policy, and selected
  beam should match the stable path before timing wins are accepted.
- If `experimentalGpuKvCache=true` is combined with `numBeams > 1`, `bestOf > 1`,
  or `temperature > 0`, the executor should reject that combination rather than
  silently changing semantics.
- If `temperature > 0` is requested, decode should use sampling rather than beam
  search even if a beam size is also present.
- Temperature fallback should visibly try the requested temperatures in order
  when a quality gate rejects a segment.
- `language: "auto"` should not silently bias to English when splitgraph
  decoder-init logits identify a different language token. Browser validation
  for this may require a non-English fixture.

## Report Template

Please report:

- Harness commit/branch and how it was linked to this working tree.
- Browser/runtime used.
- Exact URL or command for each run.
- Model preset and flags.
- Transcript text or token prefix.
- Key metrics: preprocess, encode, decode, total, RTFx, decoder step p50/p95.
- Whether GPU tensor downloads remain zero on the greedy GPU-KV path.
- Any console errors or thrown guardrail errors.
- Verdict: pass, fail, or inconclusive, with the smallest repro if failing.

## Batched Beam Status

Stable beam remains the correctness oracle. The opt-in batched beam path is now
implemented and has one paired browser pass, but it should stay experimental
until it is checked against more model variants, beam sizes, EOS/timestamp cases,
and longer audio.
