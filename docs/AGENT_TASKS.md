# Agent Task Coordination

Branch: `feat/asr-pipeline-output-formats`
Updated: 2026-05-30 (Flexo-DSV4Pro — comprehensive redesign)

## BLOCKER CHAIN (read top-to-bottom)

```
CTC refactor (src/ctc/) ──────► WAV2VEC2 model (Phase E) ──► WAV2VEC2 alignment (Phase D2)
                                       │
                                       └──► WAV2VEC2 ASR (Phase E)
```

**Rule**: Phase D2 blocked until Phase E. Phase E blocked until CTC refactor.
Everything else can proceed in parallel.

---

## READY NOW — No Blockers

### T1: VAD Backend Adapters — ASSIGNED (Flexo-DSV4Pro)

Priority: HIGH (enables actual VAD-based chunking)
Dependencies: NONE (runtime VAD already exists)
Files:
- `src/chunking/backends/ten-vad.ts` — TenVAD adapter implementing WhisperVadBackend
- `src/chunking/backends/firered-vad.ts` — FireRed VAD adapter
- Tests: `tests/chunking-backends.test.ts`

### T2: Fixed Window Chunker — ASSIGNED (Flexo-DSV4Pro)

Priority: MEDIUM (fallback when VAD unavailable)
Dependencies: NONE
Files:
- `src/chunking/fixed-window.ts` — 30s window, 28s hop, 2s overlap
- Tests: `tests/chunking-fixed-window.test.ts`

### T3: Post-Processing Extras — ASSIGNED (Flexo-DSV4Pro)

Priority: MEDIUM
Dependencies: NONE
Files:
- `src/post-processing/word-deduplicator.ts` — cross-window word dedup
- `src/post-processing/text-normalizer.ts` — casing, punctuation normalization
- `src/post-processing/sentence-boundary.ts` — punctuation-based sentence detection
- `src/post-processing/transcript-formatter.ts` — canonical transcript output
- Tests: `tests/post-processing-extras.test.ts`

### T4: Cross-Attention DTW Extractor — AVAILABLE

Priority: LOW (Whisper-specific, extract from existing code)
Dependencies: NONE (read from whisper-seq2seq/attention-alignment.ts)
Files:
- `src/alignment/cross-attention-dtw.ts` — extract DTW from whisper-seq2seq
- Tests: `tests/alignment-dtw.test.ts`

---

## BLOCKED — Waiting for CTC Refactor

### T5: CTC Module Refactor — ASSIGNED (Flexo-glm5.1)

Priority: HIGH (blocks WAV2VEC2 + alignment)
Status: BLOCKED on architecture decision
Files:
- `src/ctc/types.ts` — CtcLogits, CtcDecoder interface
- `src/ctc/decoder.ts` — argmaxAndSelectedLogProbs + ctcCollapseWithSpans + timing
- Migrate from `src/models/lasr-ctc/ctc.ts`
- Tests: `tests/ctc-decoder.test.ts`

### T6: WAV2VEC2 Model Completion — ASSIGNED (Flexo-glm5.1)

Priority: HIGH (blocks alignment)
Status: PARTIAL — types/config/tokenizer/ort/executor DONE (commit `94ceb99`)
Remaining:
- `src/models/wav2vec2/model.ts` — model factory
- `src/presets/wav2vec2/` — model presets (wav2vec2-base-960h, xlsr-turkish, etc.)
- WAV2VEC2 feature extractor (mel.ts — currently empty stub)
- Unit tests + smoke test with real ONNX model
- HF upload: wav2vec2-base-960h + xlsr-turkish

### T7: CTC Viterbi Forced Alignment — BLOCKED

Priority: HIGH (enables WhisperX-style alignment)
Dependencies: CTC decoder interface (T5) + WAV2VEC2 model (T6)
Files:
- `src/alignment/ctc-viterbi.ts` — forward Viterbi + backtrack
- `src/alignment/word-merger.ts` — char alignment → word boundaries
- `src/alignment/post-processor.ts` — monotonic enforcement, gap handling, clamping
- Tests: `tests/alignment-ctc-viterbi.test.ts`

### T8: WAV2VEC2 Aligner — BLOCKED

Priority: HIGH (WhisperX's key advantage: 20ms alignment)
Dependencies: CTC Viterbi (T7) + WAV2VEC2 model (T6)
Files:
- `src/alignment/wav2vec2-aligner.ts` — Wav2Vec2ForcedAligner class
- `src/alignment/models/registry.ts` — language → model mapping
- `src/alignment/models/loader.ts` — ONNX model loading
- Tests: `tests/alignment-wav2vec2.test.ts`

---

## DONE

### Completed by Flexo-DSV4Pro

| Phase | Module | Tests | Commits |
|-------|--------|-------|---------|
| Whisper Core | vanilla decode loop + onTokenLogits | 78 | `1efddda`-`7c85cdb` |
| Phase A | `src/quality/` (7 files) | 13 | `5474991` |
| Phase B | `src/chunking/` (4 files, partial) | 11 | `0ec5fba` |
| Phase C | `src/post-processing/` (2 files, partial) | 4 | `5d31de4` |
| Phase F | enhanced-executor wiring | — | `bdfef2a` |
| Phase G | package.json exports | — | `1e3edfc` |

### Completed by Flexo-glm5.1

| Phase | Module | Commits |
|-------|--------|---------|
| Phase E | `src/models/wav2vec2/` (6 files, partial) | `94ceb99` |

---

## Per-Language WAV2VEC2 Models (Reference)

| Language | HF Model | Size | Use Case |
|----------|----------|------|----------|
| English | `facebook/wav2vec2-base-960h` | 95M | ASR + alignment |
| Turkish | `m3hrdadfi/wav2vec2-large-xlsr-turkish` | 317M | Turkish alignment |
| Multi-53 | `facebook/wav2vec2-large-xlsr-53` | 300M | 53 languages |
| Multi-128 | `facebook/wav2vec2-xls-r-300m` | 300M | 128 languages |
| English Large | `facebook/wav2vec2-large-960h` | 317M | Best English alignment |

**WhisperX pattern**: "Transcribe with Whisper → align with WAV2VEC2"
- Whisper gives best accuracy across 99 languages
- WAV2VEC2 gives 20ms word timestamps (vs Whisper's ~100ms DTW)
- Best of both worlds, composable modules

---

## Shared Files (coordinate before modifying)

- `src/models/lasr-ctc/ctc.ts` — CTC decode logic (T5 will migrate this to src/ctc/)
- `src/types/index.ts` — shared type definitions
- `src/audio/specs.ts` — audio processor specs
- `src/inference/descriptors.ts` — encoder/decoder descriptors
- `package.json` — coordinate exports changes
- `src/index.ts` — coordinate barrel exports

## Communication

Leave notes in this file when claiming tasks or changing shared files.
Commit this file when updating task status.
