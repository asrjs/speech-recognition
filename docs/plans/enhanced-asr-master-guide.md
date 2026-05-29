# Speech-Recognition Enhanced ASR: Master Implementation Guide

Unified reference connecting all architecture plans, production techniques, and implementation phases.

Source documents:
- `docs/references/whisper-production-techniques.md` — hallucination suppression, alignment, long audio
- `docs/plans/whisper-vanilla-enhanced-architecture.md` — vanilla + enhanced split
- `docs/plans/whisper-enhanced-implementation-plan.md` — 11-phase plan, VAD integration
- `docs/plans/standalone-nlp-alignment-modules.md` — model-agnostic modules
- `docs/plans/wav2vec2-model-and-alignment.md` — WAV2VEC2 dual-purpose model

---

## 1. The Problem: Vanilla Whisper Hallucinates

Whisper's autoregressive decoder, when given encoder features that are mostly silence,
generates repetitive plausible-sounding text. This is the #1 quality problem.

```
Vanilla Whisper (fixed 30s window):
  [0s---speech 5-15s---SILENCE---30s]
  Decoder fills silence with: "...Thank you for watching. Thank you for watching."
```

WhisperX solves this with VAD pre-segmentation. The decoder only ever sees speech.
Our TenVAD + FireRed VAD achieve the same effect — no new dependencies needed.

**Impact breakdown:** VAD pre-segmentation = 70% of quality improvement.
Disabled context conditioning = 20%. Everything else = 10%.

---

## 2. Architecture: Layered Enhancement

```
Layer 0: core.ts                    (DONE — other agent)
  Pure vanilla decode loop, ONNX-agnostic

Layer 1: executor.ts                (DONE — other agent)
  ONNX bridge, WhisperOnnxExecutor

Layer 2: src/quality/               (NEW — model-agnostic)
  Hallucination suppression: compression ratio, logprob, entropy, temperature fallback
  Works with ANY autoregressive ASR model

Layer 3: src/chunking/              (NEW — model-agnostic)
  VAD-based audio pre-segmentation: TenVAD + FireRed VAD backends
  Drift handler, overlap handler, fixed-window chunker
  Works with ANY ASR model

Layer 4: src/post-processing/       (NEW — model-agnostic)
  Segment merger, word deduplication, sentence boundary, text normalization
  Works with ANY ASR output

Layer 5: src/alignment/             (NEW — model-agnostic)
  CTC Viterbi forced alignment (from WAV2VEC2 logits)
  Cross-attention DTW (from Whisper attention weights)
  Works with ANY transcript + audio pair

Layer 6: src/models/wav2vec2/       (NEW — dual-purpose ASR model)
  WAV2VEC2 as standalone CTC ASR model
  Same ONNX graph also used for forced alignment
  Reuses lasr-ctc/ctc.ts CTC decode pipeline
```

---

## 3. Module Dependency Graph

```
                    ┌─────────────┐
                    │  src/types/  │  ← shared types
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────────┐
            │              │                  │
     ┌──────┴──────┐  ┌───┴────┐  ┌──────────┴──────────┐
     │ src/quality  │  │chunking│  │ src/post-processing  │
     │ (no deps)   │  │(runtime│  │ (no deps)            │
     │             │  │  VAD)  │  │                      │
     └──────┬──────┘  └───┬────┘  └──────────┬───────────┘
            │              │                  │
            └──────────────┼──────────────────┘
                           │
                 ┌─────────┴─────────┐
                 │  src/alignment/    │
                 │  (CTC Viterbi,     │
                 │   DTW, WAV2VEC2)   │
                 └─────────┬─────────┘
                           │
            ┌──────────────┼──────────────────┐
            │              │                  │
   ┌────────┴────────┐  ┌─┴──────────┐  ┌───┴──────────────┐
   │ Whisper Enhanced │  │ Parakeet   │  │ WAV2VEC2         │
   │ Executor         │  │ Enhanced   │  │ (ASR + alignment)│
   │ (wraps vanilla)  │  │ Executor   │  │                  │
   └─────────────────┘  └────────────┘  └──────────────────┘
```

Each module is independently importable:
- `@asrjs/speech-recognition/quality`
- `@asrjs/speech-recognition/chunking`
- `@asrjs/speech-recognition/post-processing`
- `@asrjs/speech-recognition/alignment`

---

## 4. Quality Gates: Complete Reference

All pure functions. No ONNX dependency. Work with any AR decoder.

### 4.1 Compression Ratio (faster-whisper, whisper.cpp)

```
ratio = len(raw_bytes) / len(zlib_compress(raw_bytes))
threshold: 2.4
"the the the the" → ~4.0 (hallucinated)
"Hello world"     → ~1.2 (normal)
```

Implementation: pako (already in dependency tree).

### 4.2 Average Log Probability (faster-whisper, whisper.cpp)

```
For each generated token:
  log_prob = logits[chosen] - log_sum_exp(logits)
avg_logprob = mean(log_probs)
threshold: -1.0
Good: ~-0.3  |  Hallucinated: ~-2.0
```

Requires: logit collection from decode loop (request via handoff doc).

### 4.3 No-Speech Probability (faster-whisper, whisper.cpp)

```
no_speech_prob = softmax(first_token_logits)[50362]
Dual check: no_speech_prob > 0.6 AND avg_logprob < -1.0
→ Skip segment entirely (silence)
```

Whisper-specific (token 50362). VAD handles this better for non-Whisper models.

### 4.4 Entropy Filter (whisper.cpp)

```
Per token: H = -sum(p * log(p)) over vocabulary
avg_entropy = mean(entropy_per_token)
threshold: 2.4 nats
Confident: H ≈ 0.5  |  Uncertain: H ≈ 10.0
```

Model-agnostic. Any model with logit output.

### 4.5 Temperature Fallback (faster-whisper, whisper.cpp)

```
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for temp in temperatures:
  result = transcribe(audio, temperature=temp)
  if quality_gates.accept(result): return result
  if quality_gates.no_speech(result): return empty
return last_result
```

Higher temperature breaks repetitive loops. Model-agnostic.

### 4.6 VAD Pre-Segmentation (WhisperX's secret weapon)

```
1. Run VAD on full audio → speech segments
2. Pad each segment by 400ms
3. Cap at 29s (under Whisper's 30s window)
4. Feed only speech segments to ASR model
5. Never process silence → never hallucinate
```

**70% of WhisperX's quality improvement.** Model-agnostic.

---

## 5. Long Audio: Stitching & Drift

### 5.1 Drift Handler (from whisper.cpp)

```
seek = 0 (in mel frames)
For each 30s window:
  segments = decode(window)
  For each segment:
    absolute_start = seek + segment.start
    absolute_end = seek + segment.end
    corrected_start = max(absolute_start, seek)  // can't go backwards
  advance seek to max(seek + 100, last_segment_end)  // min 1s, max 30s
```

Prevents cumulative timestamp drift. Model-agnostic pattern.

### 5.2 Segment Stitching

```
Window 1: [2.0-5.5s] "Hello" [5.5-12.8s] "world test"
Window 2: [28.5-35.2s] "of the" [35.2-42.1s] "emergency"

Final: merge all → adjust offsets → deduplicate overlapping words
```

Post-processing module handles this. Model-agnostic.

---

## 6. Alignment: Two Methods

### 6.1 Cross-Attention DTW (Whisper native)

```
1. Extract cross-attention from alignment heads during decode
2. Average across heads → [tokens × audio_frames]
3. Argmax per token → approximate audio position
4. DTW enforces monotonicity
Resolution: ~100ms
No extra model needed
```

Whisper-specific. Already implemented in `src/models/whisper-seq2seq/`.

### 6.2 CTC Viterbi Forced Alignment (WAV2VEC2)

```
1. Run WAV2VEC2 on audio → per-frame char probabilities [T × V]
2. Tokenize transcript to character IDs
3. Build CTC expanded path: [blank, t0, blank, t1, blank, t2, ..., blank]
4. Viterbi forward: alpha[t][s] = max(stay, advance, skip) + emission
5. Backtrack → frame index per character
6. Group chars → words → word timestamps
Resolution: ~20ms (50fps)
Requires: WAV2VEC2 model (per language)
```

Model-agnostic algorithm. WAV2VEC2 is just the feature extractor.
The Viterbi code works with ANY model that produces CTC logits.

---

## 7. WAV2VEC2: Dual-Purpose Model

Same ONNX graph, two inference modes:

### ASR Mode
```
audio → WAV2VEC2 features → ONNX encoder → CTC logits
→ argmax + CTC collapse (reuse lasr-ctc/ctc.ts) → transcript + word timestamps
```

### Alignment Mode
```
audio + transcript → WAV2VEC2 features → ONNX encoder → CTC logits
→ CTC Viterbi forced alignment → frame-accurate word timestamps
```

### What gets shared with existing code

| Component | Already exists in | WAV2VEC2 reuses |
|-----------|-------------------|-----------------|
| CTC argmax + log probs | `lasr-ctc/ctc.ts` | Yes, as-is |
| CTC collapse with spans | `lasr-ctc/ctc.ts` | Yes, as-is |
| Frame → seconds timing | `lasr-ctc/ctc.ts` | Yes, as-is |
| Sentence timing | `lasr-ctc/ctc.ts` | Yes, as-is |
| CTC Viterbi algorithm | New: `alignment/ctc-viterbi.ts` | Shared across models |
| Feature extraction | New: `wav2vec2/mel.ts` | WAV2VEC2-specific |

### Popular models on HuggingFace

```
English:   facebook/wav2vec2-base-960h (95M), facebook/wav2vec2-large-960h (317M)
Turkish:   m3hrdadfi/wav2vec2-large-xlsr-turkish
Multi-53:  facebook/wav2vec2-large-xlsr-53 (300M)
Multi-128: facebook/wav2vec2-xls-r-300m, facebook/wav2vec2-xls-r-1b
```

### ONNX export: much simpler than Whisper

Single graph, no KV cache, no autoregressive decoder, no 4-graph split.
One forward pass: encoder → CTC logits. ~360MB fp32, ~180MB fp16, ~90MB q8.

---

## 8. How Models Use These Modules

### Whisper Enhanced

```ts
const audio = loadAudio('meeting.wav');

// 1. VAD chunking (model-agnostic)
const segments = await segmentAudio(audio, 16000, { backend: 'ten-vad' });

// 2. Per-chunk: vanilla Whisper + quality gates + temperature fallback
const results = [];
for (const seg of segments) {
  const result = await withTemperatureFallback(
    (temp) => whisper.transcribe(seg.audio, { temperature: temp }),
    [compressionRatioGate(2.4), logProbGate(-1.0)],
  );
  results.push({ ...result, offset: seg.start });
}

// 3. Merge (model-agnostic)
const merged = mergeSegments(results);
const deduped = deduplicateWords(merged);

// 4. Optional: WAV2VEC2 alignment (replaces Whisper's DTW)
const aligner = new Wav2Vec2ForcedAligner({ model: 'wav2vec2-base-en' });
const aligned = await aligner.align(audio, deduped.text);
```

### Parakeet TDT Enhanced

```ts
// Same modules, different model
const segments = await segmentAudio(audio, 16000, { backend: 'firered-vad' });
for (const seg of segments) {
  const result = await parakeet.transcribe(seg.audio);
  const quality = evaluateGates(result, [compressionRatioGate(2.4)]);
  // ...
}
```

### WAV2VEC2 Standalone ASR

```ts
const model = await createModel('wav2vec2-base-en');
const result = await model.transcribe(audio);
// Frame-accurate word timestamps built-in
```

### Alignment Only (with any model's transcript)

```ts
const whisper = await createModel('whisper-large-v3-turbo');
const transcript = await whisper.transcribe(audio);

const aligner = new Wav2Vec2ForcedAligner({ model: 'wav2vec2-base-en' });
const aligned = await aligner.align(audio, transcript.text);
// 20ms-resolution word timestamps
```

---

## 9. Implementation Phases (Unified)

### Phase A: `src/quality/` — Hallucination Suppression (no deps)

New files:
```
src/quality/types.ts
src/quality/compression-ratio.ts       ← pako
src/quality/log-probability.ts         ← logSumExp
src/quality/entropy.ts                 ← Shannon entropy
src/quality/no-speech.ts               ← token 50362
src/quality/temperature-fallback.ts    ← retry loop
src/quality/evaluator.ts               ← composite runner
src/quality/index.ts
```
Tests: `tests/quality-*.test.ts`

### Phase B: `src/chunking/` — VAD Segmentation (depends on runtime VAD)

New files:
```
src/chunking/types.ts
src/chunking/backends/ten-vad.ts       ← adapter for TenVAD
src/chunking/backends/firered-vad.ts   ← adapter for FireRed VAD
src/chunking/vad-segmenter.ts          ← VAD → speech segments
src/chunking/fixed-window.ts           ← 30s sliding window
src/chunking/drift-handler.ts          ← seek counter
src/chunking/overlap-handler.ts        ← window overlap
src/chunking/index.ts
```
Tests: `tests/chunking-*.test.ts`

### Phase C: `src/post-processing/` — Transcript Refinement (no deps)

New files:
```
src/post-processing/types.ts
src/post-processing/segment-merger.ts
src/post-processing/word-deduplicator.ts
src/post-processing/text-normalizer.ts
src/post-processing/sentence-boundary.ts
src/post-processing/transcript-formatter.ts
src/post-processing/index.ts
```
Tests: `tests/post-processing-*.test.ts`

### Phase D: `src/alignment/` — Forced Alignment (depends on ONNX)

New files:
```
src/alignment/types.ts
src/alignment/ctc-viterbi.ts           ← model-agnostic CTC Viterbi
src/alignment/cross-attention-dtw.ts   ← extract from whisper-seq2seq
src/alignment/word-merger.ts           ← char alignment → words
src/alignment/post-processor.ts        ← monotonic, gaps, clamping
src/alignment/models/registry.ts       ← language → model mapping
src/alignment/models/loader.ts         ← WAV2VEC2 ONNX loading
src/alignment/wav2vec2-aligner.ts      ← WAV2VEC2 alignment pipeline
src/alignment/index.ts
```
Tests: `tests/alignment-*.test.ts`

### Phase E: `src/models/wav2vec2/` — Dual-Purpose ASR Model

New files:
```
src/models/wav2vec2/types.ts
src/models/wav2vec2/config.ts
src/models/wav2vec2/mel.ts             ← WAV2VEC2 feature extractor
src/models/wav2vec2/tokenizer.ts       ← CTC character vocab
src/models/wav2vec2/ort.ts             ← ONNX session
src/models/wav2vec2/executor.ts        ← reuse lasr-ctc/ctc.ts
src/models/wav2vec2/model.ts           ← factory
src/models/wav2vec2/index.ts
src/presets/wav2vec2/                  ← model presets
```
Tests: `tests/wav2vec2-*.test.ts`

### Phase F: Wire Enhanced Executors

```
src/models/whisper-seq2seq/enhanced-executor.ts   ← wraps vanilla
src/models/nemo-tdt/enhanced-executor.ts           ← wraps parakeet (future)
src/models/lasr-ctc/enhanced-executor.ts           ← wraps medasr (future)
```

### Phase G: ONNX Export Tool (for WAV2VEC2)

```
tools/wav2vec2-onnx-export/export_wav2vec2.py     ← single-graph export
```

---

## 10. Package Exports

```json
{
  "exports": {
    ".": "...",
    "./quality": { "types": "./dist/quality/index.d.ts", ... },
    "./chunking": { "types": "./dist/chunking/index.d.ts", ... },
    "./post-processing": { "types": "./dist/post-processing/index.d.ts", ... },
    "./alignment": { "types": "./dist/alignment/index.d.ts", ... }
  }
}
```

---

## 11. Key Rules

1. **quality/, chunking/, post-processing/, alignment/ must NOT import from src/models/**
2. **Each module is independently importable** — zero coupling
3. **quality/ and post-processing/ have ZERO external dependencies** (pure TS)
4. **chunking/ depends only on existing runtime VAD** (TenVAD, FireRed)
5. **alignment/ depends on ONNX Runtime** (for WAV2VEC2 model loading)
6. **wav2vec2/ reuses lasr-ctc/ctc.ts** — shared CTC decode pipeline
7. **Do NOT touch executor.ts, core.ts, processors.ts** — other agent's domain
8. **Follow TDD** — tests first, then implementation
9. **Gate:** typecheck + lint + test + build after each phase

---

## 12. Reference Document Map

| Document | What it covers | Lines |
|----------|----------------|-------|
| `docs/references/whisper-production-techniques.md` | Hallucination, alignment, long audio, quality gates | 467 |
| `docs/plans/whisper-vanilla-enhanced-architecture.md` | Vanilla + enhanced split, feature catalog, API | 273 |
| `docs/plans/whisper-enhanced-implementation-plan.md` | 11-phase plan, VAD integration, handoff | 624 |
| `docs/plans/standalone-nlp-alignment-modules.md` | Model-agnostic module architecture | 375 |
| `docs/plans/wav2vec2-model-and-alignment.md` | WAV2VEC2 ASR + alignment dual purpose | 407 |
| **This document** | **Unified master guide** | **~350** |
