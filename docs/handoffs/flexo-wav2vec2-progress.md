# Flexo — WAV2VEC2 CTC Model Progress

Branch: `feat/asr-pipeline-output-formats`
Last updated: 2026-05-31
Author: Flexo (glm-5.1, home P520)

## Commit History (WAV2VEC2 + CTC related)

```
PENDING: refactor: extract shared CTC module, migrate wav2vec2 + lasr-ctc executors
bdfef2a refactor: wire enhanced-executor to standalone quality/chunking/post-processing modules (Phase F)
5d31de4 refactor: extract post-processing/ module (Phase C)
0ec5fba refactor: extract chunking/ module (Phase B)
5474991 refactor: extract quality/ module (Phase A)
94ceb99 feat: add WAV2VEC2 CTC model family — types, config, tokenizer, ORT glue, executor
```

## Phase 1: WAV2VEC2 Model Files (commit 94ceb99) — DONE

| File | Lines | Description |
|------|-------|-------------|
| `src/models/wav2vec2/types.ts` | 206 | All interfaces: config, artifacts, transcription options, native output, executor |
| `src/models/wav2vec2/config.ts` | 107 | DEFAULT_WAV2VEC2_CONFIG, parseWav2Vec2Config, describeWav2Vec2Model |
| `src/models/wav2vec2/tokenizer.ts` | 127 | Wav2Vec2CharTokenizer — char-level CTC vocab, encode/decode |
| `src/models/wav2vec2/ort.ts` | 227 | ONNX Runtime glue — session creation, artifact resolution, HF download |
| `src/models/wav2vec2/executor.ts` | ~670 | OrtWav2Vec2Executor — full inference pipeline (deduplicated buildWords) |
| `src/models/wav2vec2/index.ts` | 2 | Barrel exports |

## Phase 2: CTC Module Extraction — DONE (pending commit)

### New Files

| File | Lines | Description |
|------|-------|-------------|
| `src/ctc/types.ts` | ~170 | Generic CTC types: CtcTokenSpan, CtcUtteranceTiming, CtcSentenceTiming, CtcNativeWord, CtcDecoderConfig, CtcDecodeResult |
| `src/ctc/decoder.ts` | ~420 | CtcDecoder class + stateless functions: argmax, collapse, timing, sentences, words |
| `src/ctc/index.ts` | ~25 | Barrel exports |
| `tests/ctc-decoder.test.ts` | ~460 | 25 tests: parity, word building, CtcDecoder class, backward compat |

### Migration

1. `src/ctc/decoder.ts` — new shared module (functions moved from lasr-ctc/ctc.ts)
2. `src/models/lasr-ctc/ctc.ts` — thin re-export wrapper (backward compat)
3. `src/models/lasr-ctc/executor.ts` — imports from `../../ctc/index.js` (was `./ctc.js`)
4. `src/models/wav2vec2/executor.ts` — imports from `../../ctc/index.js` (was `../lasr-ctc/ctc.js`)
5. Removed `buildWordsFromSpans()` from wav2vec2/executor.ts — uses shared `buildWordsFromCharSpans()`
6. Removed `Wav2Vec2NativeWord` unused import

### Type Aliases (backward compat)

```ts
// lasr-ctc/ctc.ts exports:
export type { CtcTokenSpan as LasrCtcTokenSpan } from '../../ctc/types.js';
export type { CtcUtteranceTiming as LasrCtcUtteranceTiming } from '../../ctc/types.js';
export type { CtcSentenceTiming as LasrCtcSentenceTiming } from '../../ctc/types.js';
```

### Architecture

```
src/ctc/
  types.ts      — Generic types (CtcTokenSpan, CtcUtteranceTiming, etc.)
                  No model-specific imports. Pure contract.
  decoder.ts    — CtcDecoder class + stateless functions
                  Stateless: argmaxAndSelectedLogProbs, ctcCollapseWithSpans,
                  estimateSecondsPerOutputFrame, addTimesToTokenSpans,
                  buildUtteranceTiming, buildSentenceTimings, buildWordsFromCharSpans
                  Class: CtcDecoder(blankId, vocabSize, tokenizer, wordSeparator?)
                    .decodeFromLogits() → CtcDecodeResult (full pipeline)
                    .argmax(), .collapse(), .addTiming(), .buildUtterance(),
                    .buildSentences(), .buildWords() (individual steps)
  index.ts      — barrel exports

src/models/lasr-ctc/
  ctc.ts        — RE-EXPORT WRAPPER ONLY (backward compat)
  executor.ts   — imports from ../../ctc/index.js
  types.ts      — LasrCtcTokenSpan etc. still defined locally (unchanged)

src/models/wav2vec2/
  executor.ts   — imports from ../../ctc/index.js, no local buildWordsFromSpans
  types.ts      — Wav2Vec2TokenSpan etc. still defined locally (unchanged)
```

### Models Using CTC

| Model | blankId | vocab | Word Strategy | Frame Rate |
|-------|---------|-------|---------------|------------|
| MedASR (lasr-ctc) | 0 (epsilon) | BPE sentencepiece | No auto word building (BPE handles) | varies |
| WAV2VEC2 base-960h | 0 (pad) | 32 char-level | buildWordsFromCharSpans(' ') | 49/sec |
| Future CTC models | TBD | TBD | TBD | TBD |

## ONNX Model

- Source: `facebook/wav2vec2-base-960h`
- File: `/tmp/wav2vec2-base-960h.onnx` (1.71 MB, opset 18)
- Input: `input_values` [batch, seq] float32 (raw waveform)
- Output: `logits` [batch, frames, 32] float32
- 49 frames/sec at 16kHz (outputStride=320)
- Vocab: 32 tokens — 26 uppercase letters + pad(0), <s>(1), </s>(2), <unk>(3), |(4=space), '(5)
- CTC blank token ID: 0 (`<pad>`)

## Gate Status (after CTC refactor)

- typecheck: ✓ clean
- lint: ✓ 0 errors
- tests: 547/549 pass (2 pre-existing flaky whisper tests, unrelated)
  - CTC-specific: 25/25 new + 3/3 legacy = 28/28 ✓
- build: ✓ clean

## Remaining Work

### Next: Model Factory + Presets
- W2V-3: `src/models/wav2vec2/model.ts` — model factory
- W2V-3: `src/presets/wav2vec2/` — preset manifest + factory
- W2V-3: `src/models/wav2vec2/mapping.ts` — native → canonical transcript

### After Model Factory
- W2V-4: Smoke test with real ONNX model (`/tmp/wav2vec2-base-960h.onnx`)
- HF upload: `wav2vec2-base-960h` to HuggingFace
- CTC Viterbi forced alignment (`src/alignment/ctc-viterbi.ts`)
- Final gate check

### Future (deferred)
- Deduplicate Wav2Vec2TokenSpan → CtcTokenSpan alias in wav2vec2/types.ts
- Deduplicate LasrCtcTokenSpan → CtcTokenSpan alias in lasr-ctc/types.ts
- Mixed dtype
- q4/q4f16

## Other Agent's Work

- Flexo-DSV4Pro completed Phases A/B/C/F (quality/chunking/post-processing extraction, enhanced-executor wiring)
  - Commits: 5474991 → bdfef2a
  - Files owned: `src/quality/`, `src/chunking/`, `src/post-processing/`, enhanced executor files
  - Do NOT modify without coordination

## Environment Notes

- Python venv: `tools/whisper-onnx-export/.venv/`
- Python 3.12, ONNX Runtime 1.26.0
- ONNX export at `/tmp/wav2vec2-base-960h.onnx` (may need re-export if /tmp cleaned)
