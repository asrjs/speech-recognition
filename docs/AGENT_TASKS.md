# Agent Task Coordination

Branch: `feat/asr-pipeline-output-formats`
Updated: 2026-05-31 (Flexo-glm5.1)

## Context Recovery

**Progress file**: `docs/handoffs/flexo-wav2vec2-progress.md`
Read this file first after any context reset. Contains:
- Complete commit history for WAV2VEC2 + CTC work
- All completed files with line counts
- ONNX model details (location, specs, vocab)
- CTC module architecture
- Remaining work checklist

## Dependency Chain

```
src/ctc/ (shared CTC module) ✅ DONE ──┐
                                        ├──► WAV2VEC2 model factory (Phase E) ──► Alignment (Phase D)
                                        │
quality/ (Phase A) ✅ DONE              │
chunking/ (Phase B) ✅ DONE             │
post-processing/ (Phase C) ✅ DONE      │
                                        │
enhanced-executor (Phase F) ✅ DONE ────┘
```

## COMPLETED TASKS

### CTC Module Extraction — DONE (Flexo-glm5.1)

**New shared module: `src/ctc/`**
- `src/ctc/types.ts` — Generic CTC types (CtcTokenSpan, CtcUtteranceTiming, etc.)
- `src/ctc/decoder.ts` — CtcDecoder class + stateless functions
- `src/ctc/index.ts` — barrel exports
- `tests/ctc-decoder.test.ts` — 25 tests (parity + word building + class)

**Migration:**
- `lasr-ctc/ctc.ts` → re-export wrapper (backward compat preserved)
- `lasr-ctc/executor.ts` → imports from `../../ctc/index.js`
- `wav2vec2/executor.ts` → imports from `../../ctc/index.js`, removed local buildWordsFromSpans

**Gate:** typecheck ✓, lint ✓, 547/549 tests pass, build ✓

### Whisper Vanilla Core — DONE (Flexo-DSV4Pro)
Commit: `1efddda`

### Whisper Enhanced Modules — DONE (Flexo-DSV4Pro)
Commits: `708aac9` through `7c85cdb`

### Phases A/B/C/F (quality/chunking/post-processing/enhanced) — DONE (Flexo-DSV4Pro)
Commits: `5474991` through `bdfef2a`

### T1-T3: VAD Backends + Fixed-Window + Post-Extras — DONE (Flexo-DSV4Pro)
Commit: `1db314c`
- TenVAD + FireRed backend adapters (chunking/backends/)
- FixedWindowChunker (30s/28s/2s overlap)
- Post-processing extras: deduplicateWords, normalizeText, buildSentences
- 558 tests pass

### T4: Cross-Attention DTW Extractor — DONE (delegate_task, Flexo-DSV4Pro)
Commit: pending
- `src/alignment/cross-attention-dtw.ts` — crossAttentionDtwTimestamps()
- `src/alignment/index.ts`, `src/alignment.ts` entry stub
- `tests/alignment-dtw.test.ts` — 5 tests
- Package.json export: `@asrjs/speech-recognition/alignment`
- 563 tests pass

### WAV2VEC2 Model Files — DONE (Flexo-glm5.1)
Commit: `94ceb99`
- types.ts, config.ts, tokenizer.ts, ort.ts, executor.ts, index.ts

## ACTIVE TASKS

### Phase E: WAV2VEC2 — IN PROGRESS (Flexo-glm5.1)

Owner: Flexo-glm5.1
Started: 2026-05-30
Status: CTC module done, model factory next
Progress file: `docs/handoffs/flexo-wav2vec2-progress.md`

Files owned (do not modify without coordination):
- `src/ctc/**` (new shared module)
- `src/models/wav2vec2/**`
- `tests/ctc-decoder.test.ts`
- `src/presets/wav2vec2/**` (not yet created)

Remaining:
- [ ] Model factory: `src/models/wav2vec2/model.ts`
- [ ] Native→canonical mapping: `src/models/wav2vec2/mapping.ts`
- [ ] Presets: `src/presets/wav2vec2/`
- [ ] Unit tests for model factory
- [ ] Smoke test with real ONNX model
- [ ] HF upload: wav2vec2-base-960h

### Phase D: src/alignment/ — PARTIAL (DTW DONE, CTC Viterbi BLOCKED)

Owner: Flexo-DSV4Pro (DTW), UNASSIGNED (CTC Viterbi)
Dependencies: WAV2VEC2 model (Phase E) for CTC Viterbi
Status: DTW ✅, CTC Viterbi ⏳

Completed:
- `src/alignment/cross-attention-dtw.ts` — crossAttentionDtwTimestamps()
- `src/alignment/index.ts`, `src/alignment.ts`, package.json export
- 5 tests

Remaining:
- `src/alignment/ctc-viterbi.ts` — will use CtcDecoder (from src/ctc/)
- `src/alignment/wav2vec2-aligner.ts` — WAV2VEC2 alignment backend
- Tests: `tests/alignment-ctc-viterbi.test.ts`

## SHARED FILES (coordinate before modifying)

- `src/models/lasr-ctc/ctc.ts` — RE-EXPORT WRAPPER, delegates to `src/ctc/`
- `src/models/lasr-ctc/types.ts` — LasrCtcTokenSpan etc. (should become aliases later)
- `src/types/index.ts` — shared type definitions
- `src/audio/specs.ts` — audio processor specs
- `src/inference/descriptors.ts` — encoder/decoder descriptors
- `package.json` — coordinate exports changes
- `src/index.ts` — coordinate barrel exports

## Communication

Leave notes in this file when claiming tasks or changing shared files.
Commit this file when updating task status.
