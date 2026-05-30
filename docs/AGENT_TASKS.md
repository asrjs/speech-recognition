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
                                        │                                             └──► Alignment (Phase D) ✅ DONE
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

### Phase D: src/alignment/ — DONE
Owner: Flexo (DSV4Pro/gpt-5.5)
Commits: `33cc27b` (ctc-viterbi), `4ab5e89` (wav2vec2-aligner), `eb2cc6d` (token label/separator hardening)
Status: ✅ Complete — 24 focused tests (15 CTC Viterbi + 9 WAV2VEC2 aligner)

Completed:
- [x] `src/alignment/ctc-viterbi.ts` — `ctcForceAlign()`, `ctcViterbiBacktrack()`, `ctcLogSoftmax()`
- [x] `src/alignment/wav2vec2-aligner.ts` — `createWav2Vec2Aligner()`, `groupCharAlignmentToWords()`
- [x] Tests: `tests/alignment-ctc-viterbi.test.ts` (15), `tests/wav2vec2-alignment.test.ts` (10)
- [x] Token label callback: `ctcForceAlign(..., { tokenToChar })` so Wav2Vec2 separator tokens decode to spaces before word grouping.
- [x] `createWav2Vec2AlignerFromLogits()` for executor-extracted/reusable logits.
- [x] Vitest source alias for `@asrjs/speech-recognition/alignment` so alignment tests exercise `src/`, not stale `dist/`.

### WAV2VEC2 Real ONNX Forced Alignment Smoke — DONE
Owner: Flexo (gpt-5.5)
Commit: pending
Status: ✅ Complete

Implemented:
- [x] `OrtWav2Vec2Executor.extractLogits()` exposes reusable `[frames, vocab]` logits, tokenizer, blank ID, duration, and shape metadata.
- [x] `tests/smoke/wav2vec2-node-wasm-align-smoke.mjs` runs real Node/WASM ONNX logits extraction and CTC Viterbi forced alignment.
- [x] Verified on `tests/fixtures/jfk2.en.wav` with `/tmp/wav2vec2-base-960h.onnx`.

Observed forced-alignment smoke:

```text
wav2vec2 node/wasm forced-alignment smoke passed
frames=549 vocab=32 chars=104 words=22
and:0.42-0.48 so:0.64-0.82 my:1.02-1.16 fellow:1.32-1.56 americans:1.70-2.18 ask:3.55-3.69
```

### Beam Search — DONE
Owner: Flexo
Commits: `a0bdb9e` (core.ts), `d2ce555` (executor.ts), `783dfd1` (types.ts)
Status: ✅ Complete

Completed:
- [x] `core.ts`: `whisperDecode()` dispatch, `whisperBeamDecode()` (beam search with KV-cache-per-beam)
- [x] `executor.ts`: `splitGraphDecodeLoop` accepts `numBeams` + `lengthPenalty`, dispatches to `whisperDecode`
- [x] `types.ts`: `numBeams`, `lengthPenalty`, `patience`, `bestOf` params
- [x] WhisperX param parity: beam_size, best_of, patience, length_penalty

### ORT URL/Path Fix — DONE
Owner: Flexo
Commit: `87e5e6a`
Status: ✅ Complete

Fixed:
- [x] `tokenizer.ts:fetchText()` now handles bare file paths in Node.js (not just `file://` URLs)
- [x] Verified: `WhisperTokenizer.fromUrl('/tmp/.../tokenizer.json')` works
- [x] `ort.ts:createWhisperOrtSession()` already handled both — documented
- [ ] Remaining: `loadSpeechModel` direct-source path has wiring issue at `materializeHuggingFaceArtifacts` (manipulates URLs even for `kind='direct'`). Workaround: use direct session creation pattern.

### Phase E: Splitgraph Bridge Fix — DONE
Owner: Flexo
Commit: `cef005f`
Status: ✅ Complete

Root cause: `transcribeWithSplitGraph` bridge had three bugs:
1. `dims: []` — tensors without shape (ORT rejection)
2. Prefix mismatch — init outputs `present.*`, step expects `past_key_values.*`
3. Encoder KV lost — step model outputs only decoder KV

Fixed:
- [x] runInit: store tensor dims for both `present.*` and `past_key_values.*` prefixes
- [x] runStep: reconstruct `ort.Tensor` with proper dims + prefix conversion + data cloning
- [x] runDecoderStepSplit: clone tensor data for cross-session safety

Verified: whisper-base fp32, end-of-chapter-4.en.mp3 (167s)
→ 73.6s transcription, 84.8% word overlap, zero hallucinations

### Phase F: Whisper Production Checklist — ACTIVE
Owner: Flexo
Status: ✅ Core works | ⏳ Features pending

Checklist:
- [x] Greedy decode (argmax per step)
- [x] Beam search (whisperBeamDecode, numBeams, lengthPenalty)
- [x] Long audio windowing (transcribeWithWindowing, 30s windows)
- [x] Timestamp tokens (WhisperTimestampLogitProcessor)
- [x] Token suppression (suppressTokens, beginSuppressTokens from generation_config)
- [x] Context conditioning (extraPromptTokens, condition_on_previous_text)
- [x] Quality gates (compression, logprob, entropy, no-speech)
- [x] Temperature fallback [0.0, 0.2, ..., 1.0]
- [x] Mel dimension auto-detect (numMelBins from manifest)
- [x] SRT/VTT subtitle export
- [x] VRAM optimization (skip merged decoder, defer alignment)
- [x] URL/path unified (fetchText handles bare paths + file://)
- [ ] Large-v3-turbo smoke test (needs fp32/fp16 on 8GB+ or use q8 with ORT fix)
- [ ] bestOf independent decodings (multiple runs, pick best)
- [ ] patience (beam search early stopping)
- [ ] Diarization (separate model, future Phase)
- [ ] Batched encoder for parallel window processing (WhisperX-style)

### WAV2VEC2 follow-ups — UNASSIGNED

Remaining:
- HF upload/publish for the ONNX Wav2Vec2 base-960h artifact.
- Optional npm script for the Wav2Vec2 smoke commands if this becomes recurring.
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
npx vitest run tests/wav2vec2-model.test.ts tests/wav2vec2-alignment.test.ts
npm run typecheck
npm run lint          # 0 errors, existing max-lines warnings only
npm test              # 103 files, 601 tests passed
npm run build
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
node tests/smoke/wav2vec2-node-wasm-align-smoke.mjs
```

## Communication

Leave notes in this file when claiming tasks or changing shared files. Commit this file when updating task status.
