# Agent Task Coordination

Branch: `feat/asr-pipeline-output-formats`
Updated: 2026-05-30 (Flexo-DSV4Pro)

## Dependency Chain (CRITICAL)

```
CTC decode (lasr-ctc/ctc.ts) ──┐
                                ├──► WAV2VEC2 model (Phase E) ──► Alignment (Phase D)
                                │
quality/ (Phase A) ──┐         │
chunking/ (Phase B) ─┤─────────┼──► Enhanced Executors (Phase F)
post-processing/ (Phase C) ───┘
```

**Rule**: Phase D blocked until Phase E done. Phase E blocked until CTC verify pass.
Phases A/B/C have NO blockers — relocate existing code from `src/models/whisper-seq2seq/`.

## ACTIVE TASKS

### Phase A: src/quality/ — ASSIGNED (Flexo-DSV4Pro)

Owner: Flexo-DSV4Pro
Started: 2026-05-30
Status: STARTING
Dependencies: NONE

Scope:
- Move quality gates from `src/models/whisper-seq2seq/` to `src/quality/`
- Files: enhanced-types → quality/types.ts, quality-gates → split into compression-ratio/log-probability/entropy/no-speech/evaluator, temperature-fallback → quality/temperature-fallback.ts
- Tests: `tests/quality-*.test.ts`
- Add `./quality` export to package.json
- Re-export from whisper-seq2seq/index.ts for backward compat

### Phase B: src/chunking/ — ASSIGNED (Flexo-DSV4Pro)

Owner: Flexo-DSV4Pro  
Dependencies: Phase A complete
Status: AFTER PHASE A

Scope:
- Move: vad-segmenter → chunking/vad-segmenter.ts, drift-handler → chunking/drift-handler.ts
- Add: chunking/backends/ten-vad.ts, chunking/backends/firered-vad.ts (adapters)
- Tests: `tests/chunking-*.test.ts`
- Add `./chunking` export to package.json

### Phase C: src/post-processing/ — ASSIGNED (Flexo-DSV4Pro)

Owner: Flexo-DSV4Pro
Dependencies: Phase A complete
Status: AFTER PHASE A

Scope:
- Move: segment-merger → post-processing/segment-merger.ts
- Add: post-processing/word-deduplicator.ts, text-normalizer.ts, sentence-boundary.ts
- Tests: `tests/post-processing-*.test.ts`
- Add `./post-processing` export to package.json

### Phase E: src/models/wav2vec2/ — CLAIMED (Flexo-glm5.1)

Owner: Flexo-glm5.1 (other instance)
Started: 2026-05-30
Status: IN PROGRESS
Dependencies: lasr-ctc/ctc.ts CTC decode (SHARED, read-only)

Scope:
- `src/models/wav2vec2/` — new model family (CTC ASR)
- `src/presets/wav2vec2/` — model presets
- Tests: `tests/wav2vec2-*.test.ts`
- ONNX export tool: `tools/wav2vec2-onnx-export/`

Files owned (do not modify without coordination):
- `src/models/wav2vec2/**`
- `src/presets/wav2vec2/**`
- `tests/wav2vec2-*.test.ts`

### Phase D: src/alignment/ — BLOCKED

Owner: UNASSIGNED
Dependencies: Phase E complete (WAV2VEC2 model)
Status: BLOCKED — waiting for WAV2VEC2

Scope:
- `src/alignment/ctc-viterbi.ts` — CTC forced alignment algorithm
- `src/alignment/wav2vec2-aligner.ts` — WAV2VEC2 alignment backend
- `src/alignment/cross-attention-dtw.ts` — extracted from whisper-seq2seq
- `src/alignment/post-processor.ts` — monotonic enforcement, gap handling
- Tests: `tests/alignment-*.test.ts`

## COMPLETED TASKS

### Whisper Vanilla Core — DONE (Flexo-DSV4Pro)

Commit: `1efddda`
- `core.ts` — pure decode loop (ONNX-agnostic)
- `executor.ts` — ONNX bridge (delegates to core)

### Whisper Enhanced Modules (8 phases) — DONE (Flexo-DSV4Pro)

Commits: `708aac9` through `7c85cdb`
- 8 source files + 8 test files in `src/models/whisper-seq2seq/`
- 78 new tests, 489 total pass
- EnhancedWhisperExecutor wraps WhisperExecutor via composition
- **These files will be RELOCATED in Phases A/B/C**

## SHARED FILES (coordinate before modifying)

- `src/models/lasr-ctc/ctc.ts` — CTC decode logic (imported by MedASR + WAV2VEC2)
- `src/types/index.ts` — shared type definitions
- `src/audio/specs.ts` — audio processor specs
- `src/inference/descriptors.ts` — encoder/decoder descriptors
- `package.json` — coordinate exports changes
- `src/index.ts` — coordinate barrel exports

## Communication

Leave notes in this file when claiming tasks or changing shared files.
Commit this file when updating task status.
