# Agent Task Coordination

Branch: `feat/asr-pipeline-output-formats`
Updated: 2026-05-31 (Flexo-gpt5.5)

## Context Recovery

**Progress file**: `docs/handoffs/flexo-wav2vec2-progress.md`
Read this file first after any context reset. Contains:
- Completed WAV2VEC2 + shared CTC files
- Node/WASM smoke command and known local ONNX model location
- Remaining alignment/HF-upload work

## Dependency Chain

```
src/ctc/ (shared CTC module) ✅ DONE ──┐
                                        ├──► WAV2VEC2 model factory/preset ✅ DONE ──► real ONNX smoke ✅ DONE
                                        │                                             └──► Alignment (Phase D) next
quality/ (Phase A) ✅ DONE              │
chunking/ (Phase B) ✅ DONE             │
post-processing/ (Phase C) ✅ DONE      │
                                        │
enhanced-executor (Phase F) ✅ DONE ────┘
```

## COMPLETED TASKS

### CTC Module Extraction — DONE (Flexo)

**Shared module: `src/ctc/`**
- `src/ctc/types.ts` — Generic CTC types (CtcTokenSpan, CtcUtteranceTiming, etc.)
- `src/ctc/decoder.ts` — CtcDecoder class + stateless functions
- `src/ctc/index.ts` — barrel exports
- `tests/ctc-decoder.test.ts` — 25 tests (parity + word building + class)

**Migration:**
- `lasr-ctc/executor.ts` imports from `../../ctc/index.js`
- `wav2vec2/executor.ts` imports from `../../ctc/index.js`, removed local buildWordsFromSpans
- `lasr-ctc/ctc.ts` currently remains a re-export wrapper; backward compatibility is no longer a requirement and can be removed when MedASR is rewritten.

### WAV2VEC2 Model Files — DONE

- `src/models/wav2vec2/types.ts`
- `src/models/wav2vec2/config.ts`
- `src/models/wav2vec2/tokenizer.ts`
- `src/models/wav2vec2/ort.ts`
- `src/models/wav2vec2/executor.ts`

### WAV2VEC2 Model Factory + Presets — DONE (Flexo-gpt5.5)
Commits: `b25e9a6`, `64c1cea`

Implemented:
- `src/models/wav2vec2/model.ts` — `createWav2Vec2ModelFamily()` with stub fallback and real `OrtWav2Vec2Executor` when a source is provided.
- `src/models/wav2vec2/mapping.ts` — native Wav2Vec2 CTC transcript → canonical `TranscriptResult` mapping.
- `src/models/wav2vec2/index.ts`, `src/models/wav2vec2.ts` — model subpath exports.
- `src/presets/wav2vec2/manifest.ts` — `facebook/wav2vec2-base-960h` manifest + aliases.
- `src/presets/wav2vec2/factory.ts` — `createWav2Vec2PresetFactory()`.
- `src/presets/wav2vec2/index.ts`, `src/presets/wav2vec2.ts`, `src/presets/index.ts` — preset exports.
- `src/runtime/builtins.ts` — registers wav2vec2 model family and preset.
- `src/presets/descriptors.ts` — built-in descriptor for Wav2Vec2 CTC raw-waveform model.
- `tests/wav2vec2-model.test.ts` — model family, preset, built-in registration coverage.
- `tests/preset-descriptors.test.ts` — Wav2Vec2 catalog metadata coverage.

### WAV2VEC2 Real ONNX Node/WASM Smoke — DONE (Flexo-gpt5.5)
Commits: `b25e9a6`, `64c1cea`

Implemented:
- `tests/smoke/wav2vec2-node-wasm-smoke.mjs`
- Direct local source support with explicit external data:
  - `modelUrl: file:///tmp/wav2vec2-base-960h.onnx`
  - `modelDataUrl: file:///tmp/wav2vec2-base-960h.onnx.data`
  - `modelDataFilename: wav2vec2-base-960h.onnx.data`

Smoke command:

```bash
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
```

Observed output:

```text
wav2vec2 node/wasm smoke passed
sampleRate=16000 duration=11.000s elapsed≈8.6s
words=22 tokens=105
and so my fellow americans ask not what your country can do for you ask what you can do for your country
```

### Whisper Vanilla Core — DONE
Commit: `1efddda`

### Whisper Enhanced / Quality / Chunking / Post-processing / Alignment DTW — DONE
Commits: `5474991` through `1744585`.

## ACTIVE TASKS

### Phase D: src/alignment/ — PARTIAL (DTW DONE, CTC Viterbi next)

Owner: UNASSIGNED
Dependencies: WAV2VEC2 model + smoke are now green.
Status: DTW ✅, WAV2VEC2 ASR ✅, CTC Viterbi ⏳

Remaining:
- `src/alignment/ctc-viterbi.ts` — use `src/ctc/` logits/path utilities.
- `src/alignment/wav2vec2-aligner.ts` — Wav2Vec2 alignment backend.
- Tests: `tests/alignment-ctc-viterbi.test.ts` and Wav2Vec2 aligner fixtures.

### WAV2VEC2 follow-ups — UNASSIGNED

Remaining:
- HF upload/publish for the ONNX Wav2Vec2 base-960h artifact.
- Optional npm script for the Wav2Vec2 smoke command if this becomes recurring.
- Remove `lasr-ctc/ctc.ts` compatibility wrapper when MedASR is rewritten.

## SHARED FILES (coordinate before modifying)

- `src/ctc/**` — shared CTC decoder/timing utilities.
- `src/models/wav2vec2/**` — Wav2Vec2 CTC model family.
- `src/presets/wav2vec2/**` — Wav2Vec2 preset manifest/factory.
- `src/types/index.ts` — shared type definitions.
- `src/audio/specs.ts` — audio processor specs.
- `src/inference/descriptors.ts` — encoder/decoder descriptors.
- `package.json` — coordinate exports changes.
- `src/index.ts` — coordinate barrel exports.

## Verification (latest W2V pass)

```bash
npx vitest run tests/wav2vec2-model.test.ts tests/preset-descriptors.test.ts tests/exports.test.ts
npm run typecheck
npm run lint          # 0 errors, existing max-lines warnings only
npm test              # 100 files, 568 tests passed
npm run build
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
```

## Communication

Leave notes in this file when claiming tasks or changing shared files. Commit this file when updating task status.
