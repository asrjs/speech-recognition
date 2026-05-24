# FireRed VAD Degraded Troubleshooting

This note documents a real failure mode seen while resuming `streaming-demo` work in May 2026.

## Symptoms

In `streaming-demo`, warnings appeared:

- `FireRed VAD is degraded. Runtime plots may be incomplete.`
- `Speech segmentation is running on the rough gate. FireRed VAD is diagnostics-only.`

And browser console sometimes showed:

- `NotFoundError: A requested file or directory could not be found ...`

## What Was Actually Broken

There were two overlapping issues:

1. **IndexedDB cache fragility** under stale/migrated stores:
   - cache open/transaction paths could throw `NotFoundError`
   - this could degrade FireRed VAD init even when model URLs were valid
2. **CMVN URL expectation drift**:
   - FireRed defaults previously pointed at a remote `cmvn.json` location that can be absent
   - worker init did not forward `cmvnJsonUrl` into packed runtime create options

## Why This Was Confusing

- `Speech segmentation is running on the rough gate...` is expected in the current Parakeet-style detector port when VAD diagnostics are enabled.
- So users could see both:
  - one expected architecture warning
  - one true runtime failure warning

## Fix Applied

### 1) Make cache failures fail-open

Files:

- `src/io/cache.ts`
- `src/runtime/firered-vad/core/asset-cache.ts`
- `src/runtime/firered-vad/core/loader.ts`

Changes:

- recover from cross-realm `NotFoundError` by `error.name` matching
- reset/reopen DB state when store is stale
- never block model loading if cache read/write fails

### 2) Stabilize CMVN defaults and wiring

Files:

- `src/runtime/firered-vad-browser.ts`
- `src/runtime/firered-vad-worker.ts`
- `src/runtime/firered-vad/api/classes.ts`

Changes:

- default CMVN now resolves to bundled asset URL:
  - `new URL('./firered-vad/assets/cmvn.json', import.meta.url).href`
- worker forwards `cmvnJsonUrl` into `FireredVadStreamPacked.create(...)`
- packed runtime options propagate `cmvnJsonUrl` into `modelUrls.cmvnJsonUrl`

### 3) Surface real degrade reason

File:

- `src/runtime/streaming-detector.ts`

Change:

- degraded warning now appends backend error detail when available

## How To Verify

1. Start `streaming-demo` and select FireRed backend.
2. Start mic capture.
3. Confirm no `NotFoundError` in browser console during init.
4. Confirm warning banner does not show degraded state.
5. If degraded appears, use appended error detail in warning text to triage quickly.

## Agent Checklist (When This Reappears)

1. Check warning text detail (now includes backend error if present).
2. Check browser console for `NotFoundError` and asset fetch failures.
3. Verify FireRed model URLs:
   - `fireredvad_stream_vad_with_cache.onnx`
   - `fireredvad_vad.onnx`
   - `fireredvad_aed.onnx`
4. Confirm CMVN resolves to bundled asset unless explicitly overridden.
5. Re-test with clean dev server after dependency or alias changes.

