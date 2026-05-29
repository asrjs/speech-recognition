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

### Whisper Executor Refactoring — OTHER AGENT

Owner: Other agent (opencode/codex)
Files: `src/models/whisper-seq2seq/core.ts`, `executor.ts`, `processors.ts`
Status: IN PROGRESS (commit `1efddda`)

Do not touch these files.

### Whisper Enhanced Modules — NOT YET CLAIMED

Files: `src/quality/`, `src/chunking/`, `src/post-processing/`
Plan: `docs/plans/enhanced-asr-master-guide.md`
Status: WAITING for vanilla executor to stabilize

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
