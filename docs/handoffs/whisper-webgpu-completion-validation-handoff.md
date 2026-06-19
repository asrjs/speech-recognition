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
Vitest: 112 files passed, 1 skipped; 649 tests passed, 4 skipped
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

Final metrics: total `12672.18ms`, transcribe `12671.84ms`, preprocess
`78.46ms`, encode `378.855ms`, decode `12188.81ms`, RTFx `2.3629`,
decoder init `95.76ms`, decoder step `10385.625ms`, step p50 `105.985ms`,
step p95 `117.67ms`, step count `98`, GPU tensor downloads `0`, KV location
`cpu`.

## Browser Harness To Test

Fast test harness:

```text
N:\github\asrjs\webgpu-agent-test
```

Recommended validation order:

1. Sync or point the harness at the local `speech-recognition` working tree.
2. Run the default greedy WebGPU path with `experimentalGpuKvCache=true`.
3. Run the stable beam path with GPU-KV disabled.
4. Run a temperature-fallback scenario with GPU-KV disabled, since GPU-KV is
   intentionally greedy/temperature-0 only.

## Expected Behavior

- Greedy WebGPU GPU-KV remains the fast path and should still produce the known
  JFK transcript prefix without token-policy regressions.
- Beam search should run on the stable splitgraph path. It should not use
  `experimentalGpuKvCache`.
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

## Do Not Test As A Performance Win Yet

Do not treat beam search as optimized. Current beam work is correctness-first.
Batched beam decode is still future work and should be compared against this
stable path before any timing claims are accepted.
