# WAV2VEC2 as First-Class ASR Model + Alignment Backend

## Implementation Status (2026-05-30)

DONE:
- `src/alignment/ctc-viterbi.ts` — model-agnostic CTC Viterbi forced alignment.
- `src/alignment/wav2vec2-aligner.ts` — Wav2Vec2 transcript-to-word alignment backend over CTC Viterbi.
- `createWav2Vec2AlignerFromLogits()` — reuses executor-extracted logits without second ONNX pass.
- Public exports from `@asrjs/speech-recognition/alignment`.
- Focused tests: `tests/alignment-ctc-viterbi.test.ts` (15) + `tests/wav2vec2-alignment.test.ts` (10).
- Regression: Wav2Vec2 separator tokens are decoded to spaces via `tokenToChar` before word grouping.
- Real ONNX forced-alignment smoke: `tests/smoke/wav2vec2-node-wasm-align-smoke.mjs` (JFK fixture, 22 words, 549 frames, monotonic timestamps).

NEXT:
- HF upload/publish for Wav2Vec2 ONNX artifact.
- WebGPU smoke for Wav2Vec2 (Node/WASM is validated).

## The Insight

WAV2VEC2 is a CTC model. It outputs per-frame character probabilities.
Two inference modes from the same ONNX graph:

1. **ASR mode:** argmax + CTC collapse → transcript with word timestamps
2. **Alignment mode:** CTC Viterbi forced alignment → frame-accurate word timestamps

We already have CTC decoding infrastructure in `src/models/lasr-ctc/ctc.ts`:
- `argmaxAndSelectedLogProbs()` — frame-level argmax + log probabilities
- `ctcCollapseWithSpans()` — CTC collapse (merge repeats, remove blanks) with timing
- `addTimesToTokenSpans()` — frame indices → seconds
- `buildSentenceTimings()` — sentence-level segmentation from punctuation

WAV2VEC2 can reuse ALL of this. The only differences from MedASR:
- Different mel frontend (WAV2VEC2 uses its own feature extractor, not kaldi-mel)
- Different tokenizer (character-level, not sentencepiece)
- Different model architecture (WAV2VEC2 encoder, not conformer)
- Same CTC output → same decode pipeline

## Dual-Purpose Architecture

```
                    ┌──────────────────────────────────┐
                    │   WAV2VEC2 ONNX Model            │
                    │   (single graph: encoder → CTC)   │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────┴───────────┐
                    │                      │
            ASR Mode                Alignment Mode
                    │                      │
        ┌───────────┴────────┐  ┌─────────┴──────────────┐
        │ argmax + CTC       │  │ CTC Viterbi             │
        │ collapse           │  │ forced alignment        │
        │ (reuse ctc.ts)     │  │ (new: ctc-viterbi.ts)   │
        └───────────┬────────┘  └─────────┬──────────────┘
                    │                      │
        transcript + word        aligned word timestamps
        timestamps               (for any ASR model's
                                 transcript)
```

## File Structure

```
src/models/wav2vec2/                     ← NEW model family
  types.ts          — model config, artifact sources, transcript types
  config.ts         — model classification, config parsing
  mel.ts            — WAV2VEC2 feature extractor (25ms window, 20ms hop, 80 mel)
  ort.ts            — ONNX session creation + artifact resolution
  tokenizer.ts      — character-level CTC vocabulary + decode
  model.ts          — model wiring + factory
  executor.ts       — Wav2Vec2Executor (ASR mode)
  index.ts          — barrel exports

src/alignment/
  ctc-viterbi.ts    — CTC forced alignment algorithm (shared)
  wav2vec2-aligner.ts — WAV2VEC2 alignment backend (uses same ONNX session)
  ...

src/models/lasr-ctc/
  ctc.ts            — EXISTING, shared CTC decode logic (reuse as-is)
```

## WAV2VEC2 ASR Model

### Model Config

```ts
interface Wav2Vec2ModelConfig {
  readonly ecosystem: 'meta';           // Meta/Facebook
  readonly architecture: 'wav2vec2';
  readonly processorArchitecture: 'wav2vec2-fe';  // WAV2VEC2 feature extractor
  readonly encoderArchitecture: 'wav2vec2-transformer';
  readonly decoderArchitecture: 'ctc';
  readonly sampleRate: number;          // 16000
  readonly featureHopMs: number;        // 20ms (50fps)
  readonly featureWindowMs: number;     // 25ms
  readonly nMels: number;              // could be raw waveform or mel
  readonly vocabularySize?: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;    // character-level CTC vocab
  readonly ctcBlankId: number;          // usually 0
}
```

### Executor (ASR Mode)

```ts
class Wav2Vec2Executor {
  async transcribe(audio: AudioBufferLike, options): Promise<Wav2Vec2Transcript> {
    // 1. Feature extraction (WAV2VEC2-specific)
    const features = this.extractFeatures(audio);

    // 2. Run ONNX encoder → CTC logits [1, T, V]
    const { logits, vocabSize, frameCount } = await this.runInference(features);

    // 3. CTC decode (reuse from lasr-ctc/ctc.ts)
    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(
      logits, frameCount, vocabSize,
    );
    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(
      frameIds, selectedLogProbs, config.ctcBlankId,
    );

    // 4. Timing
    const secondsPerFrame = estimateSecondsPerOutputFrame({
      audioDurationSec: audioDuration,
      outFrames: frameCount,
    });
    const timedSpans = addTimesToTokenSpans(tokenizer, tokenSpans, secondsPerFrame);

    // 5. Decode text
    const text = tokenizer.decode(collapsedIds);
    const sentences = buildSentenceTimings(text, tokenizer, collapsedIds, timedSpans);

    // 6. Also expose raw logits for alignment mode
    return { text, words: wordTimestamps, sentences, logits, frameCount, vocabSize };
  }
}
```

This is almost identical to MedASR's executor. The CTC decode pipeline is shared.
Only the frontend (feature extraction) and tokenizer differ.

### Why WAV2VEC2 is a Good ASR Model

| Model | Accuracy | Speed | Timestamps | Languages |
|-------|----------|-------|------------|-----------|
| Whisper large-v3-turbo | Very high | Medium | Cross-attention DTW (~100ms) | 100+ |
| Parakeet TDT | Very high | Fast | TDT native (~20ms) | English |
| MedASR CTC | High | Fast | CTC frame-level (~20ms) | English |
| **WAV2VEC2** | **High** | **Fast** | **CTC frame-level (~20ms)** | **Per-language models** |

WAV2VEC2 advantages:
- Frame-accurate timestamps (50fps = 20ms resolution) — same as MedASR
- Can also do forced alignment (unique dual-purpose)
- Per-language fine-tuned models available on HuggingFace
- Works in browser via ONNX Runtime Web
- Good for alignment use case (WhisperX uses it specifically for this)

### Popular WAV2VEC2 Models on HuggingFace

```
English:
  facebook/wav2vec2-base-960h          — 95M params, base model
  facebook/wav2vec2-large-960h         — 317M params, large model
  facebook/wav2vec2-large-robust-ft    — robust, noise-tolerant

Turkish:
  m3hrdadfi/wav2vec2-large-xlsr-turkish
  mpoyraz/wav2vec2-large-xlsr-53-turkish

Multilingual:
  facebook/wav2vec2-large-xlsr-53      — 53 languages, 300M params
  facebook/wav2vec2-xls-r-300m         — 128 languages
  facebook/wav2vec2-xls-r-1b           — 128 languages, 1B params
```

## Alignment Mode (Dual Purpose)

The same WAV2VEC2 ONNX model used for ASR can also be used for forced alignment:

```ts
// In src/alignment/wav2vec2-aligner.ts
import type { AlignedWord, AlignmentResult } from './types.js';

export class Wav2Vec2ForcedAligner {
  // Can accept a Wav2Vec2Executor's session (shared) or create its own
  constructor(private readonly session: Wav2Vec2OnnxSession) {}

  async align(
    audio: Float32Array,
    transcript: string,
    language?: string,
  ): Promise<AlignmentResult> {
    // 1. Feature extraction (same as ASR mode)
    const features = extractFeatures(audio);

    // 2. Run ONNX → CTC logits [1, T, V]
    const { logits, vocabSize, frameCount } = await this.session.run(features);

    // 3. Tokenize transcript to character IDs (same CTC vocab)
    const targetTokens = this.tokenizeTranscript(transcript);

    // 4. CTC Viterbi forced alignment
    const alignment = ctcForceAlign(
      logits,           // [T, V] frame-level log probabilities
      targetTokens,     // [N] character-level target tokens
      this.config.ctcBlankId,
    );

    // 5. Convert character alignment → word timestamps
    const words = charAlignmentToWords(alignment, transcript, frameCount);

    // 6. Post-process (monotonic, gaps, clamping)
    return postProcessAlignment(words, audioDuration);
  }
}
```

### CTC Viterbi Algorithm (shared module)

```ts
// In src/alignment/ctc-viterbi.ts — model-agnostic

export interface CtcAlignmentResult {
  readonly charFrames: ReadonlyArray<{
    char: string;
    frame: number;       // frame index
    seconds: number;     // time in seconds
    confidence: number;  // alignment score
  }>;
}

export function ctcForceAlign(
  logits: Float32Array,   // [T*V] flat, row-major
  frameCount: number,
  vocabSize: number,
  targets: readonly number[],  // character IDs
  blankId: number = 0,
): CtcAlignmentResult {
  // 1. Compute log-softmax
  const logProbs = logSoftmax(logits, frameCount, vocabSize);

  // 2. Build expanded CTC path: [blank, t0, blank, t1, blank, t2, ..., blank]
  const S = 2 * targets.length + 1;

  // 3. Initialize Viterbi trellis
  const alpha = new Float64Array(frameCount * S).fill(-Infinity);
  alpha[0 * S + 0] = logProbs[0 * vocabSize + blankId];      // blank at t=0
  if (S > 1) {
    alpha[0 * S + 1] = logProbs[0 * vocabSize + targets[0]];  // first char at t=0
  }

  // 4. Forward pass
  for (let t = 1; t < frameCount; t++) {
    for (let s = 0; s < S; s++) {
      const tokenIdx = s % 2 === 1 ? targets[(s - 1) / 2] : blankId;
      const emission = logProbs[t * vocabSize + tokenIdx];

      let best = alpha[(t - 1) * S + s];         // stay
      if (s > 0) best = Math.max(best, alpha[(t - 1) * S + s - 1]);  // advance
      // Skip blank between different characters
      if (s > 1 && tokenIdx !== targets[Math.floor((s - 2) / 2)]) {
        best = Math.max(best, alpha[(t - 1) * S + s - 2]);  // skip
      }

      alpha[t * S + s] = best + emission;
    }
  }

  // 5. Backtrack
  // Find best final state, then trace back through trellis
  // Returns frame index for each character

  // 6. Group characters into words
  // Convert frame indices to seconds (frameCount / audioDuration)
  return buildAlignmentResult(backtrackPath, targets, frameCount);
}
```

This algorithm is completely model-agnostic. Any model that produces CTC logits can use it.
WAV2VEC2 produces CTC logits → can use it. Any future CTC model → can use it.

## Preset Structure

```
src/presets/wav2vec2/
  index.ts          — barrel exports
  manifest.ts       — model manifest definitions
  factory.ts        — create Wav2Vec2Model from preset
  catalog.ts        — available models:
    wav2vec2-base-en        — facebook/wav2vec2-base-960h (English)
    wav2vec2-large-en       — facebook/wav2vec2-large-960h (English)
    wav2vec2-xlsr-53-tr     — m3hrdadfi/wav2vec2-large-xlsr-turkish
    wav2vec2-xlsr-53        — facebook/wav2vec2-large-xlsr-53 (multilingual)
    wav2vec2-xls-r-300m     — facebook/wav2vec2-xls-r-300m (128 languages)
```

## What Gets Shared

| Component | Source | Used by |
|-----------|--------|---------|
| CTC argmax + collapse | `lasr-ctc/ctc.ts` | MedASR, WAV2VEC2 |
| CTC timing (frames→seconds) | `lasr-ctc/ctc.ts` | MedASR, WAV2VEC2 |
| Sentence timing | `lasr-ctc/ctc.ts` | MedASR, WAV2VEC2 |
| CTC Viterbi forced alignment | `alignment/ctc-viterbi.ts` (new) | WAV2VEC2 aligner |
| Quality gates | `quality/` (new) | All models |
| VAD chunking | `chunking/` (new) | All models |
| Post-processing | `post-processing/` (new) | All models |

## How Users Use It

### Standalone ASR

```ts
import { createModel } from '@asrjs/speech-recognition';

const model = await createModel('wav2vec2-base-en');
const result = await model.transcribe(audioData);
// result.text = "Hello world"
// result.words = [{word: "Hello", start: 0.12, end: 0.54}, ...]
```

### Alignment for Another Model's Transcript

```ts
import { Wav2Vec2ForcedAligner } from '@asrjs/speech-recognition/alignment';

// Use Whisper for transcription (better accuracy)
const whisper = await createModel('whisper-large-v3-turbo');
const whisperResult = await whisper.transcribe(audioData);

// Use WAV2VEC2 for alignment (better timestamps)
const aligner = new Wav2Vec2ForcedAligner({ model: 'wav2vec2-base-en' });
const aligned = await aligner.align(audioData, whisperResult.text);
// aligned.words = [{word: "Hello", start: 0.123, end: 0.542, confidence: 0.97}, ...]
```

### Combined: ASR + Self-Alignment

```ts
const model = await createModel('wav2vec2-base-en');
const result = await model.transcribe(audioData, {
  alignment: 'self',  // use same model for alignment
});
// result has both CTC word timestamps AND forced-alignment refined timestamps
```

## Implementation Priority

### Phase 1: WAV2VEC2 as ASR model
1. `src/models/wav2vec2/types.ts`
2. `src/models/wav2vec2/config.ts`
3. `src/models/wav2vec2/mel.ts` — WAV2VEC2 feature extractor
4. `src/models/wav2vec2/tokenizer.ts` — CTC vocab
5. `src/models/wav2vec2/ort.ts` — ONNX session
6. `src/models/wav2vec2/executor.ts` — reuse `lasr-ctc/ctc.ts`
7. `src/models/wav2vec2/model.ts` — factory
8. `src/presets/wav2vec2/` — presets
9. Tests + validation

### Phase 2: CTC Viterbi alignment module
1. `src/alignment/ctc-viterbi.ts` — model-agnostic forced alignment
2. `src/alignment/wav2vec2-aligner.ts` — WAV2VEC2 alignment backend
3. Tests with known audio + transcript pairs

### Phase 3: Wire into enhanced executors
1. Whisper enhanced executor → option to use WAV2VEC2 alignment
2. Parakeet enhanced executor → option to use WAV2VEC2 alignment
3. All models → quality gates + VAD chunking + post-processing

## ONNX Export for WAV2VEC2

WAV2VEC2 models on HuggingFace are typically PyTorch.
We need ONNX export similar to what we did for Whisper.

```python
# Export script (new: tools/wav2vec2-onnx-export/export_wav2vec2.py)
from transformers import Wav2Vec2ForCTC
import torch

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# WAV2VEC2 is single-pass: encoder → CTC head
# No KV cache, no autoregressive decoder
# Much simpler than Whisper's 4-graph split
dummy_input = torch.randn(1, 80, 3000)  # [batch, features, time]

torch.onnx.export(
    model,
    dummy_input,
    "wav2vec2-base.onnx",
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch", 2: "time"},
        "logits": {0: "batch", 1: "time"},
    },
    opset_version=17,
)
```

Much simpler than Whisper — single graph, no KV cache, no split.
WAV2VEC2-base is ~360MB fp32, ~180MB fp16, ~90MB q8.

## Key Design Decision: Not a Separate Package

WAV2VEC2 is a model family inside `@asrjs/speech-recognition`, not a separate package.
Same pattern as Whisper, Parakeet, MedASR — all in one library.

The alignment functionality IS exposed as a separate import path
(`@asrjs/speech-recognition/alignment`) but uses the same model loading infrastructure.

## What This Enables

1. **Users get WAV2VEC2 as an ASR model** — fast, frame-accurate timestamps, CTC-based
2. **Users get WAV2VEC2 as an alignment backend** — force-align any transcript to audio
3. **Whisper users get WhisperX-quality alignment** — WAV2VEC2 forced alignment for word timestamps
4. **Parakeet users get alignment too** — same alignment backend for any model
5. **CTC decode pipeline shared** — MedASR and WAV2VEC2 reuse the same CTC collapse logic
6. **Single ONNX graph, dual purpose** — same model file does ASR and alignment
