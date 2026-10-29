# Active Sprint Tasks (Sprint 2026-10-17)

Active tasks derived from [deep-review-2026-10-17.md](../reports/deep-review-2026-10-17.md). Historical task logs are archived in [docs/archive/](./).

## 3-Tier Verification Gates

No task card can be marked `DONE` without satisfying the applicable gates:

- **Gate-A (Unit & Package Boundary):**
  `npm run typecheck && npm run build && npx vitest run && node tests/smoke/browser-root-import-graph.mjs && node tests/smoke/package-tarball-smoke.mjs`
  - Validates TypeScript typing, bundle build, test suite, zero-model-leak root import graph (<45 modules, 0 model files), and package export resolution.
- **Gate-B (Multi-Fixture & Public API Parity):**
  - Models must be loaded via public API (`loadSpeechModel()` or `loadSpeechModelFromLocalEntries()`), never private/internal test harness wrappers.
  - Must verify on at least 3 fixtures: short (`jfk-short.wav`), long (>20s speech), and target language or noise.
- **Gate-C (Real Browser WebGPU/WASM Evidence):**
  - Real headless Chrome WebGPU execution (`navigator.gpu` active) producing canonical receipts under `tools/data/results/canonical/`. Node-WASM runs cannot substitute for browser WebGPU evidence.

---

## Active Task Cards

### Card T1 — Root Package (`dist/index.js`) Boundary & Zero-Model-Weight Gate
- **Status:** In Progress
- **Goal:** Ensure `@asrjs/speech-recognition` root entry (`dist/index.js`) never imports any `dist/models/*` or `dist/presets/*` statically.
- **Actions:**
  1. Remove static `builtins.ts` import from `src/runtime/load.ts` (use dynamic `await import('./builtins.js')`).
  2. Remove static model family imports and top-level normalizers from `src/runtime/transcripts.ts`.
  3. Move Whisper pipeline re-exports from `src/pipeline/index.ts` to Whisper model subpath.
  4. Enforce strict zero-model check in `tests/smoke/browser-root-import-graph.mjs`.
- **Pass Criteria:** `dist/index.js` reachable local module count < 45 (`dist/models/*` = 0). Gate-A green.

### Card T2 — Runtime Wiring Fixes for `moonshine`, `qwen3-asr`, `wav2vec2`, `nemotron`, `canary`
- **Status:** Ready
- **Goal:** Resolve `ModelLoadError` and mock stub returns across built-ins and presets.
- **Actions:**
  1. Register `'moonshine'` and `'qwen3-asr'` family IDs in `builtins.ts` and `local-adapter-registry.ts`.
  2. Fix `wav2vec2` and `nemotron` presets in `builtins.ts` to set proper `options.source`.
  3. Fix `wav2vec2-tr` vs `wav2vec2` resolution collisions and `supports(undefined)` checks.
- **Pass Criteria:** Every registered preset resolves and loads via `loadSpeechModel()` without errors.

### Card T3 — Expand `owned-session` Protection to 13 Families & Fix `parakeet-ctc` Bug
- **Status:** Ready
- **Goal:** Eliminate memory leaks, disposal race conditions, and shared-disposed state bugs.
- **Actions:**
  1. Fix `src/models/parakeet-ctc/model.ts` where `session.dispose()` broke the parent model.
  2. Adopt `registerOwnedSpeechSession` helper across all 13 unhedged model families.
  3. Standardize `TranscriptResult` canonical metadata (`modelId`, `backendId`, `sampleRate`, `durationSeconds`).
  4. Fix `detail: 'sentences+words'` handling across all families.
- **Pass Criteria:** Gate-A green; `package-tarball-smoke.mjs` verifies all subpaths; no session leak.

### Card T4 — Backend Selection Integrity (Purge Fake `webgl`/`webnn` & Unified ORT Factory)
- **Status:** Ready
- **Goal:** Ensure runtime only advertises and selects real working backends (`webgpu`, `wasm`, `native-cpu`).
- **Actions:**
  1. Remove placeholder `webnn` and `webgl` from default `registerBuiltInBackends`.
  2. Explicitly define `supportedBackendIds` across all 15 model families.
  3. Stop silent fallback in `moonshine/streaming-loader.ts` when WebGPU is requested; warn visibly if falling back.
- **Pass Criteria:** Non-WebGPU environment cleanly selects `wasm` without masquerading as `webgl`.

### Card T5 — Moonshine Streaming & Qwen3-ASR fp16 Real WebGPU Validation + Disk Cleanup
- **Status:** Part 1 DONE (Disk Cleanup), Part 2 In Progress
- **Goal:** Reclaim ~10 GB disk space on `N:` drive; validate both models in real Chrome WebGPU.
- **Actions:**
  1. [DONE] Deleted 412 dead `onnx__MatMul_*` files (5.46 GB) in `models/onnx/qwen3-asr-0.6b/`.
  2. [DONE] Compacted `decoder_step_model.onnx.data` from 7.15 GB to 2.38 GB (4.77 GB freed). Verified check_model and vitest.
  3. [DONE] Deleted duplicate perf traces in `vad-demo/perf/traces/*.json` (37.5 MB).
  4. Run Playwright + Chrome WebGPU verification on `moonshine-streaming-tiny` and `Qwen3-ASR-0.6B`.
  5. Store canonical receipts in `tools/data/results/canonical/`.
- **Pass Criteria:** Real Chrome WebGPU transcripts matching oracle; disk freed > 10 GB.

### Card T6 — Neural VAD (`TEN-VAD` / `FireRedVAD`) Real Segmentation & Soft-Split
- **Status:** In Progress
- **Goal:** Wire neural VAD into real endpointing decisions; eliminate mid-word chops on max-duration.
- **Actions:**
  1. Wire `StreamingSpeechDetector.resolveEffectiveGateMode()` to return real VAD mode when ready.
  2. Use `hasRecentSpeech()` / `hasRecentSilence()` for speech boundary decisions.
  3. Implement soft-split at low-energy frame when max-duration is reached, plus continuation overlap.
  4. Prioritize local model weights in `FireRedVadAdapter`.
- **Pass Criteria:** JFK 5.33s audio captured without truncated words; `tests/streaming-detector-neural-vad.test.ts` passes.

### Card T7 — Demo Workspace Integration, vad-demo Cleanup, Repo Hygiene
- **Status:** Part 1 DONE (Repo Hygiene & Imports), Part 2 In Progress
- **Goal:** Clean repository clutter, link demos to `@asrjs/speech-recognition`, archive legacy task logs.
- **Actions:**
  1. [DONE] Cleaned untracked harness results; updated `.gitignore` for bulk test clutter.
  2. [DONE] Fixed silent early returns in `tests/nemotron-rnnt.test.ts` to use `it.skipIf`.
  3. [DONE] Fixed import paths in `examples/whisper-splitgraph-local.mjs` to `dist/`.
  4. [DONE] Supported `process.env.ASRJS_MODELS_DIR ?? 'N:/models'` in `examples/demo/vite.config.js`.
  5. [DONE] Archived `PARAKEET_VARIANT_TASKS.md` and `PRODUCT_TASKS.md` to `docs/archive/`.
  6. Connect demo apps to root workspace and consume `@asrjs/speech-recognition`.
- **Pass Criteria:** `git status` clean; demos build against published package.
