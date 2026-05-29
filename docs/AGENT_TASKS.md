# Agent Task Coordination

Branch: `feat/asr-pipeline-output-formats`
Updated: 2026-05-30

## Claimed Tasks

### WAV2VEC2 CTC Model Implementation — CLAIMED BY FLEXO

Owner: Flexo (home agent, P520 WSL2)
Started: 2026-05-30
Status: IN PROGRESS

Scope:
- `src/models/wav2vec2/` — new model family (CTC ASR)
- `src/presets/wav2vec2/` — model presets
- `src/alignment/ctc-viterbi.ts` — CTC forced alignment algorithm
- `src/alignment/wav2vec2-aligner.ts` — WAV2VEC2 alignment backend
- Tests: `tests/wav2vec2-*.test.ts`, `tests/alignment-ctc-viterbi.test.ts`
- ONNX export tool: `tools/wav2vec2-onnx-export/`

Files I own (do not modify without coordination):
- `src/models/wav2vec2/**`
- `src/presets/wav2vec2/**`
- `src/alignment/ctc-viterbi.ts`
- `src/alignment/wav2vec2-aligner.ts`
- `tests/wav2vec2-*.test.ts`
- `tests/alignment-ctc-viterbi.test.ts`

Dependencies on other agents:
- `src/models/whisper-seq2seq/**` — OTHER AGENT (do not touch)
- `src/models/lasr-ctc/ctc.ts` — SHARED (read-only, import OK)
- `src/audio/wav2vec-conv.ts` — MINE TO FILL (currently empty stub)
- `src/alignment/` cross-attention DTW — FUTURE (not starting yet)

### Whisper Executor Refactoring — OTHER AGENT (DONE, Flexo-DSV4Pro)

Owner: Flexo-DSV4Pro (home agent, P520 WSL2)
Started: 2026-05-29, Completed: 2026-05-30
Status: ✅ DONE

Completed:
- `src/models/whisper-seq2seq/core.ts` — pure decode loop (vanilla, ONNX-agnostic)
- `src/models/whisper-seq2seq/executor.ts` — ONNX bridge (unchanged, wrapper delegates to core)
- Added `onTokenLogits` callback to core for quality gates
- Commits: `1efddda` (core extraction), `708aac9` (callback + types)

### Whisper Enhanced Modules — DONE (Flexo-DSV4Pro)

Owner: Flexo-DSV4Pro (home agent, P520 WSL2)
Completed: 2026-05-30
Status: ✅ ALL 8 PHASES DONE

All new files in `src/models/whisper-seq2seq/` (zero modifications to core/executor):
- `enhanced-types.ts` — QualityVerdict, QualityGate, EnhancedDecodeOptions, VadSegmenterConfig
- `quality-gates.ts` — compression ratio (pako), logprob, entropy, no-speech gates
- `temperature-fallback.ts` — generic retry loop with escalating temperatures
- `chunk-context.ts` — ChunkContextBuilder + buildPromptWithContext()
- `drift-handler.ts` — whisper.cpp-style seek counter for long audio
- `vad-segmenter.ts` — WhisperVadBackend interface + mergeVadSegments()
- `segment-merger.ts` — mergeWhisperSegments() with word deduplication
- `enhanced-executor.ts` — EnhancedWhisperExecutor wraps WhisperExecutor via composition

Tests: 78 new tests (489 total). Commits: `708aac9` through `7c85cdb`.
Skill docs: `~/.hermes/skills/mlops/asrjs-dev/` + reference `references/whisper-enhanced-implementation.md`

### WAV2VEC2 CTC Model Implementation — CLAIMED BY FLEXO

Owner: Flexo (home agent, P520 WSL2)
Started: 2026-05-30
Status: IN PROGRESS

## Shared Files (coordinate before modifying)

- `src/models/lasr-ctc/ctc.ts` — CTC decode logic (imported by both MedASR and WAV2VEC2)
- `src/types/index.ts` — shared type definitions
- `src/audio/specs.ts` — audio processor specs
- `src/inference/descriptors.ts` — encoder/decoder descriptors
- `package.json` — coordinate exports changes
- `src/index.ts` — coordinate barrel exports

## Communication

Leave notes in this file when claiming tasks or changing shared files.
Commit this file when updating task status.
