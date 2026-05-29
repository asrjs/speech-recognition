# Standalone NLP & Alignment Modules: Architecture Plan

## Concept

Move alignment, quality gates, VAD segmentation, and post-processing out of model-specific
directories into standalone, reusable modules. Users can import and use them independently
of any specific ASR model.

```
ASR Model Output (any model) → NLP/Alignment modules → Enhanced output
                                   ↑
                          Works with Whisper, Parakeet, MedASR, anything
```

## Current Problem

```
src/models/whisper-seq2seq/         ← alignment, timestamps are HERE (Whisper-specific)
src/models/nemo-tdt/                ← no alignment
src/models/lasr-ctc/                ← no alignment
src/runtime/firered-vad/            ← VAD is HERE (not connected to models)
src/runtime/ten-vad-browser.ts      ← TenVAD is HERE (not connected to models)
```

Alignment and post-processing are locked inside the Whisper model directory.
They should be first-class modules usable with any ASR model.

## Proposed Architecture

```
src/
  alignment/                        ← NEW: standalone alignment module
    index.ts                        ← public API
    types.ts                        ← alignment types and interfaces
    ctc-viterbi.ts                  ← CTC forced alignment algorithm (from WhisperX)
    cross-attention-dtw.ts          ← DTW alignment (extracted from whisper-seq2seq)
    word-merger.ts                  ← character/frame alignment → word timestamps
    post-processor.ts               ← monotonic enforcement, gap handling, clamping
    wav2vec2-aligner.ts             ← WAV2VEC2 ONNX model wrapper
    models/                         ← alignment model management
      registry.ts                   ← language → model mapping
      loader.ts                     ← load WAV2VEC2 ONNX model

  post-processing/                  ← NEW: standalone post-processing module
    index.ts                        ← public API
    types.ts                        ← segment, word, transcript types
    segment-merger.ts               ← overlap reconciliation, deduplication
    word-deduplicator.ts            ← cross-window word dedup
    sentence-boundary.ts            ← punctuation-based sentence segmentation
    text-normalizer.ts              ← casing, punctuation, number formatting
    transcript-formatter.ts         ← format raw ASR output into canonical form

  quality/                          ← NEW: standalone quality gates module
    index.ts                        ← public API
    types.ts                        ← QualityGate interface, QualityVerdict
    compression-ratio.ts            ← text compression ratio (pako/zlib)
    log-probability.ts              ← avg log probability from logits
    entropy.ts                      ← Shannon entropy from logits
    no-speech.ts                    ← no-speech token probability
    temperature-fallback.ts         ← temperature schedule + retry loop
    evaluator.ts                    ← composite gate runner

  chunking/                         ← NEW: standalone audio chunking module
    index.ts                        ← public API
    types.ts                        ← VadSegmenterConfig, AudioSegment
    vad-segmenter.ts                ← VAD-based audio pre-segmentation
    fixed-window.ts                 ← fixed-size window chunking
    overlap-handler.ts              ← window overlap management
    drift-handler.ts                ← seek counter + drift correction
    backends/
      ten-vad.ts                    ← TenVAD adapter for VAD segmenter
      firered-vad.ts                ← FireRed VAD adapter for VAD segmenter
      silero-vad.ts                 ← (future) Silero VAD adapter

  models/                           ← EXISTING: model-specific implementations
    whisper-seq2seq/                ← uses alignment/, quality/, chunking/ as deps
    nemo-tdt/                       ← uses alignment/, quality/, chunking/ as deps
    lasr-ctc/                       ← uses alignment/, quality/, chunking/ as deps
```

## Public API

### `@asrjs/speech-recognition/alignment`

```ts
// Standalone alignment — works with any transcript + audio

import { alignWords, CrossAttentionAligner, CtcForcedAligner } from '@asrjs/speech-recognition/alignment';

// Method 1: Cross-attention DTW (needs attention weights from model)
const aligner1 = new CrossAttentionAligner({
  alignmentHeads: [{layer: 3, head: 1}, {layer: 4, head: 2}],
});
const result1 = aligner1.align(attentionWeights, tokens, text, audioDuration);

// Method 2: CTC forced alignment (needs WAV2VEC2 model + transcript)
const aligner2 = new CtcForcedAligner({
  language: 'en',  // auto-selects WAV2VEC2 model
  // OR: modelPath: '/path/to/wav2vec2.onnx'
});
const result2 = await aligner2.align(audio, transcript);

// Both return the same type:
interface AlignmentResult {
  words: AlignedWord[];
}

interface AlignedWord {
  word: string;
  start: number;        // seconds
  end: number;          // seconds
  confidence: number;   // alignment score
}
```

### `@asrjs/speech-recognition/quality`

```ts
// Standalone quality gates — works with any ASR output

import {
  compressionRatioGate,
  logProbGate,
  entropyGate,
  evaluateGates,
  withTemperatureFallback,
} from '@asrjs/speech-recognition/quality';

// Individual gates
const gate = compressionRatioGate(2.4);
const verdict = gate(text, tokens, logits, vocabSize);
// → { verdict: 'accept' | 'reject' | 'no_speech', ... }

// Composite evaluation
const result = evaluateGates(text, tokens, logits, vocabSize, [
  compressionRatioGate(2.4),
  logProbGate(-1.0),
  entropyGate(2.4),
]);

// Temperature fallback wrapper
const final = await withTemperatureFallback(
  (temp) => model.transcribe(audio, { temperature: temp }),
  [compressionRatioGate(2.4), logProbGate(-1.0)],
  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
);
```

### `@asrjs/speech-recognition/chunking`

```ts
// Standalone audio chunking — works with any ASR model

import { segmentAudio, FixedWindowChunker } from '@asrjs/speech-recognition/chunking';

// VAD-based chunking
const segments = await segmentAudio(audioData, 16000, {
  backend: 'ten-vad',           // or 'firered-vad'
  speechThreshold: 0.5,
  minSpeechDurationMs: 250,
  minSilenceDurationMs: 100,
  speechPadMs: 400,
  maxSegmentDurationMs: 29000,
});

// Fixed-window chunking (for models without VAD)
const windows = new FixedWindowChunker({
  windowDurationMs: 30000,
  hopDurationMs: 28000,         // 2s overlap
}).chunk(audioData, 16000);
```

### `@asrjs/speech-recognition/post-processing`

```ts
// Standalone post-processing — works with any ASR output

import {
  mergeSegments,
  deduplicateWords,
  formatTranscript,
} from '@asrjs/speech-recognition/post-processing';

// Merge segments from multiple windows
const merged = mergeSegments([
  { segments: window1Segments, timeOffsetSeconds: 0 },
  { segments: window2Segments, timeOffsetSeconds: 28 },
  { segments: window3Segments, timeOffsetSeconds: 56 },
]);

// Deduplicate overlapping words
const deduped = deduplicateWords(merged.words, {
  normalizeText: true,
  minOverlapRatio: 0.5,
});

// Format into canonical transcript
const formatted = formatTranscript(deduped, {
  includeTimestamps: true,
  includeConfidence: true,
  sentenceBoundary: true,
});
```

## How Models Use These Modules

### Whisper enhanced executor (wraps vanilla)

```ts
import { segmentAudio } from '../../chunking/index.js';
import { evaluateGates, withTemperatureFallback } from '../../quality/index.js';
import { CtcForcedAligner } from '../../alignment/index.js';
import { mergeSegments, deduplicateWords } from '../../post-processing/index.js';

class EnhancedWhisperExecutor {
  async transcribe(audio, options) {
    // 1. VAD chunking (model-agnostic)
    const segments = await segmentAudio(audio, 16000, { backend: 'ten-vad' });

    // 2. Per-chunk: call vanilla Whisper
    const results = [];
    for (const seg of segments) {
      const result = await withTemperatureFallback(
        (temp) => this.vanilla.transcribe(seg.audio, { temperature: temp }),
        [compressionRatioGate(2.4), logProbGate(-1.0)],
      );
      results.push({ ...result, timeOffsetSeconds: seg.startSeconds });
    }

    // 3. Merge (model-agnostic)
    const merged = mergeSegments(results);
    const deduped = deduplicateWords(merged.words);

    // 4. Optional: WAV2VEC2 alignment (model-agnostic)
    if (options.wordTimestamps && options.alignmentBackend === 'wav2vec2') {
      const aligner = new CtcForcedAligner({ language: options.language });
      return aligner.align(audio, deduped);
    }

    return deduped;
  }
}
```

### Parakeet TDT using same modules

```ts
import { segmentAudio } from '../../chunking/index.js';
import { evaluateGates } from '../../quality/index.js';
import { mergeSegments } from '../../post-processing/index.js';

class EnhancedParakeetExecutor {
  async transcribe(audio, options) {
    // Same VAD chunking, quality gates, post-processing
    // Works because modules are model-agnostic
    const segments = await segmentAudio(audio, 16000, { backend: 'firered-vad' });

    const results = [];
    for (const seg of segments) {
      const result = await this.parakeet.transcribe(seg.audio);
      const quality = evaluateGates(result.text, result.tokens, result.logits, result.vocabSize, [
        compressionRatioGate(2.4),
      ]);
      if (quality.verdict === 'accept') results.push(result);
    }

    return mergeSegments(results);
  }
}
```

## Implementation Order

### Phase A: `src/quality/` (no model dependencies, pure functions)

1. `quality/types.ts` — QualityGate interface, QualityVerdict, QualityGateResult
2. `quality/compression-ratio.ts` — pako-based compression ratio
3. `quality/log-probability.ts` — logSumExp + avg logprob
4. `quality/entropy.ts` — Shannon entropy
5. `quality/no-speech.ts` — no-speech token extraction
6. `quality/evaluator.ts` — composite gate runner
7. `quality/temperature-fallback.ts` — retry loop
8. `quality/index.ts` — public exports

Tests: `tests/quality-*.test.ts`

### Phase B: `src/chunking/` (depends on existing runtime VAD)

1. `chunking/types.ts` — VadSegmenterConfig, AudioSegment
2. `chunking/backends/ten-vad.ts` — TenVAD adapter
3. `chunking/backends/firered-vad.ts` — FireRed VAD adapter
4. `chunking/vad-segmenter.ts` — VAD-based segmenter
5. `chunking/fixed-window.ts` — fixed window chunker
6. `chunking/drift-handler.ts` — seek counter + drift correction
7. `chunking/overlap-handler.ts` — window overlap management
8. `chunking/index.ts` — public exports

Tests: `tests/chunking-*.test.ts`

### Phase C: `src/post-processing/` (no dependencies)

1. `post-processing/types.ts` — segment, word, transcript types
2. `post-processing/segment-merger.ts` — overlap reconciliation
3. `post-processing/word-deduplicator.ts` — cross-window dedup
4. `post-processing/text-normalizer.ts` — casing, punctuation
5. `post-processing/sentence-boundary.ts` — sentence detection
6. `post-processing/transcript-formatter.ts` — canonical formatting
7. `post-processing/index.ts` — public exports

Tests: `tests/post-processing-*.test.ts`

### Phase D: `src/alignment/` (depends on ONNX Runtime)

1. `alignment/types.ts` — AlignedWord, AlignmentResult, Aligner interface
2. `alignment/cross-attention-dtw.ts` — extract from whisper-seq2seq
3. `alignment/ctc-viterbi.ts` — CTC forced alignment algorithm
4. `alignment/word-merger.ts` — character→word grouping
5. `alignment/post-processor.ts` — monotonic, gaps, clamping
6. `alignment/models/registry.ts` — language→model mapping
7. `alignment/models/loader.ts` — WAV2VEC2 ONNX loading
8. `alignment/wav2vec2-aligner.ts` — WAV2VEC2 alignment pipeline
9. `alignment/index.ts` — public exports

Tests: `tests/alignment-*.test.ts`

### Phase E: Wire modules into model executors

1. Whisper enhanced executor uses quality/ + chunking/ + post-processing/ + alignment/
2. Parakeet enhanced executor uses quality/ + chunking/ + post-processing/
3. MedASR enhanced executor uses quality/ + chunking/ + post-processing/

## Package Exports

Update `package.json` exports:

```json
{
  "exports": {
    ".": { "types": "./dist/index.d.ts", ... },
    "./alignment": { "types": "./dist/alignment/index.d.ts", ... },
    "./quality": { "types": "./dist/quality/index.d.ts", ... },
    "./chunking": { "types": "./dist/chunking/index.d.ts", ... },
    "./post-processing": { "types": "./dist/post-processing/index.d.ts", ... }
  }
}
```

Users can import standalone:

```ts
// Use alignment with your own ASR model
import { CtcForcedAligner } from '@asrjs/speech-recognition/alignment';

// Use quality gates with any model
import { evaluateGates, compressionRatioGate } from '@asrjs/speech-recognition/quality';

// Use VAD chunking standalone
import { segmentAudio } from '@asrjs/speech-recognition/chunking';
```

## Key Design Decisions

1. **Modules are model-agnostic** — no imports from `src/models/*`
2. **Each module is independently importable** — `@asrjs/speech-recognition/quality`
3. **No circular dependencies** — quality → nothing, chunking → runtime VAD, alignment → ONNX, post-processing → nothing
4. **Types are shared** — common interfaces in `src/types/`
5. **Backends are pluggable** — VAD backends (TenVAD, FireRed), alignment backends (DTW, WAV2VEC2)

## What This Enables

- **Whisper users:** get WhisperX-quality output (VAD chunking + quality gates + WAV2VEC2 alignment)
- **Parakeet users:** get the same quality improvements (VAD + quality gates work identically)
- **MedASR users:** same thing
- **Library users:** use alignment, VAD, quality gates as standalone tools
- **Future models:** any new ASR model gets all these features for free
