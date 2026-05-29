# Whisper Enhanced Executor: Full Implementation Plan

## Current State

### Other agent completed (commit `1efddda`)

```
core.ts                    ← pure vanilla decode loop (ONNX-agnostic)
├── WhisperCoreSession      ← interface: runInit/runStep
├── WhisperLogitProcessor   ← callback: mutate logits before argmax
├── WhisperDecodeOptions    ← prompt, encoder, EOS, max tokens, processor
└── whisperGreedyDecode()   ← the decode loop (matches OpenAI/HF/faster-whisper)

executor.ts                ← asrjs framework glue
├── splitGraphDecodeLoop()  ← thin wrapper → whisperGreedyDecode
├── runDecoderInit()        ← ONNX bridge (decoder_init session)
├── runDecoderStepSplit()   ← ONNX bridge (decoder_step session)
├── transcribeWithSplitGraph() ← full pipeline: mel→encoder→core→segments
└── transcribeLongAudio()   ← chunking wrapper
```

### VAD ecosystem already in the project

The project already has TWO working VAD backends in `src/runtime/`:

| VAD | Files | Backend | Status |
|-----|-------|---------|--------|
| **TenVAD** | `ten-vad-browser.ts`, `ten-vad-worker.ts`, `assets/ten-vad/ten_vad.wasm` | WASM | Production-ready, streaming |
| **FireRed VAD** | `firered-vad/` (full module), `firered-vad-browser.ts`, `firered-vad-worker.ts` | ONNX | Production-ready, streaming + file-mode + AED |

The runtime already abstracts VAD via:
- `StreamingVadBackend` type: `'ten-vad' | 'firered-vad'`
- `StreamingDetector` in `streaming-detector.ts` — gate modes, chunk analysis
- `VoiceActivityProbabilityBuffer` in `vad.ts` — probability timeline
- `VoiceActivitySegment` — start/end/probability segments

**We do NOT need Silero VAD.** The project already has two better alternatives.

### parakeet.js VAD

parakeet.js has no Silero VAD files. It was mentioned in a comment but never implemented.

## Architecture: Vanilla → Enhanced → Super

```
Layer 0: core.ts (DONE)
  Pure decode loop, ONNX-agnostic, backend-agnostic

Layer 1: executor.ts (DONE)
  ONNX bridge, artifact resolution, session management
  WhisperOnnxExecutor — vanilla, reference-accurate

Layer 2: Enhanced decode features (NEW — our work)
  Quality gates, temperature fallback, condition-on-previous-text, drift handling

Layer 3: Smart chunking (NEW — uses existing VAD)
  VAD-based pre-segmentation using TenVAD or FireRed VAD
  Segment merging, padding, overlap reconciliation

Layer 4: Advanced (FUTURE)
  Batched encoder, WAV2VEC2 alignment, diarization
```

## File Structure (new files only)

```
src/models/whisper-seq2seq/
  # === EXISTING (do not touch) ===
  core.ts                 ← DONE: pure decode loop
  executor.ts             ← DONE: ONNX bridge + vanilla pipeline
  processors.ts           ← DONE: timestamp logit processor
  tokenizer.ts            ← DONE: BPE tokenizer
  ort.ts                  ← DONE: session creation
  types.ts                ← DONE: interfaces
  beam-search.ts          ← DONE: beam search skeleton
  chunking.ts             ← DONE: basic chunking
  attention-alignment.ts  ← DONE: DTW alignment
  word-timestamps.ts      ← DONE: word timestamps
  generation-config.ts    ← DONE: config parsing
  manifest.ts             ← DONE: manifest parsing
  mapping.ts              ← DONE: preset mapping
  local-file.ts           ← DONE: local file loading
  config.ts               ← DONE: model classification
  model.ts                ← DONE: model wiring
  index.ts                ← DONE: barrel exports

  # === NEW — Layer 2: Enhanced decode ===
  enhanced-types.ts       ← types for quality gates, fallback, metrics
  quality-gates.ts        ← compression ratio, logprob, entropy, no-speech
  temperature-fallback.ts ← temperature schedule + retry loop
  chunk-context.ts        ← condition-on-previous-text prompt builder
  drift-handler.ts        ← seek counter + timestamp drift correction

  # === NEW — Layer 3: Smart chunking ===
  vad-segmenter.ts        ← VAD-based audio pre-segmentation (TenVAD + FireRed)
  segment-merger.ts       ← overlap reconciliation + timestamp adjustment

  # === NEW — Layer 4: Advanced (FUTURE) ===
  # enhanced-executor.ts   ← wraps vanilla + all enhanced features
  # batched-encoder.ts     ← GPU batched encoder (future)
  # wav2vec2-aligner.ts    ← WAV2VEC2 forced alignment (future)
  # diarize.ts             ← speaker diarization (future)
```

## Implementation Phases

### Phase 1: Enhanced Types (TDD)

**File:** `src/models/whisper-seq2seq/enhanced-types.ts`

```ts
// Quality gate result
export type QualityVerdict = 'accept' | 'reject' | 'no_speech';

export interface QualityGateResult {
  verdict: QualityVerdict;
  compressionRatio?: number;
  avgLogProb?: number;
  noSpeechProb?: number;
  entropy?: number;
  reason?: string;
}

// Per-segment metrics collected during decode
export interface SegmentQualityMetrics {
  compressionRatio: number;
  avgLogProb: number;
  noSpeechProb: number;
  entropy: number;
  temperature: number;
}

// Decode result extended with quality metrics
export interface EnhancedDecodeResult {
  tokens: readonly number[];
  text: string;
  metrics: SegmentQualityMetrics;
}

// Enhanced decode options
export interface EnhancedDecodeOptions {
  // Quality thresholds (defaults from faster-whisper/whisper.cpp)
  compressionRatioThreshold?: number;  // default 2.4
  logProbThreshold?: number;           // default -1.0
  noSpeechThreshold?: number;          // default 0.6
  entropyThreshold?: number;           // default 2.4

  // Temperature fallback
  temperatureFallback?: boolean;       // default true
  temperatures?: number[];             // default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

  // Context conditioning
  conditionOnPreviousText?: boolean;   // default true
  maxContextTokens?: number;           // default maxTargetPositions / 2
}

// VAD segmenter config
export interface VadSegmenterConfig {
  backend: 'ten-vad' | 'firered-vad';
  speechThreshold?: number;            // default 0.5
  minSpeechDurationMs?: number;        // default 250
  minSilenceDurationMs?: number;       // default 100
  speechPadMs?: number;                // default 400
  maxSegmentDurationMs?: number;       // default 29000 (under 30s Whisper window)
}
```

**Test file:** `tests/whisper-enhanced-types.test.ts`
- Type compilation tests
- Default value factory tests
- Metric computation helpers

---

### Phase 2: Quality Gates (TDD)

**File:** `src/models/whisper-seq2seq/quality-gates.ts`

All pure functions. No ONNX dependency. No imports from executor.

```ts
import type { QualityGateResult, QualityVerdict } from './enhanced-types.js';

export type QualityGate = (text: string, tokens: readonly number[],
                           logits: Float32Array[], vocabSize: number) => QualityGateResult;

// 2a. Compression ratio: len(raw) / len(compressed)
// Uses pako (already in transformers.js dependency tree)
// Threshold: 2.4 (matches OpenAI/faster-whisper/whisper.cpp)
export function compressionRatioGate(threshold: number): QualityGate;

// 2b. Log probability: mean logProb across generated tokens
// Computed from logits: logProb = logits[chosenToken] - logSumExp(logits)
// Threshold: -1.0
export function logProbGate(threshold: number): QualityGate;

// 2c. No-speech: probability of token 50362 (<|nospeech|>)
// Dual check: noSpeechProb > 0.6 AND avgLogProb < -1.0
export function noSpeechGate(threshold: number, logProbThreshold: number): QualityGate;

// 2d. Entropy: Shannon entropy of logit distribution
// H = -sum(p * log(p)) where p = softmax(logits)
// Threshold: 2.4 nats (matches whisper.cpp)
export function entropyGate(threshold: number): QualityGate;

// Composite runner
export function evaluateGates(
  text: string,
  tokens: readonly number[],
  logits: Float32Array[],
  vocabSize: number,
  gates: QualityGate[],
): QualityGateResult;
```

**Dependencies:** pako for compression ratio (check if already in node_modules).

**Test file:** `tests/whisper-quality-gates.test.ts`
- Compression ratio: known text → expected ratio
- Log probability: known logits → expected avgLogProb
- No-speech: known logits at token 50362 → expected probability
- Entropy: uniform logits → high entropy, peaked logits → low entropy
- Composite: multiple gates, first reject wins

---

### Phase 3: Temperature Fallback (TDD)

**File:** `src/models/whisper-seq2seq/temperature-fallback.ts`

```ts
import type { QualityGate, QualityGateResult } from './enhanced-types.js';

export const DEFAULT_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] as const;

export interface FallbackResult<T> {
  result: T;
  temperature: number;
  attempts: number;
  gateResults: QualityGateResult[];
}

// Generic retry loop — works with any transcribe function
export async function withTemperatureFallback<T>(
  transcribeFn: (temperature: number) => Promise<{ result: T; text: string; tokens: number[]; logits: Float32Array[]; vocabSize: number }>,
  gates: QualityGate[],
  temperatures?: readonly number[],
): Promise<FallbackResult<T>>;
```

Algorithm:
1. Call transcribeFn(temperatures[0])
2. Run evaluateGates → if accept, return
3. If no_speech, return with empty text
4. If reject, try next temperature
5. After all exhausted, return last result

**Test file:** `tests/whisper-temperature-fallback.test.ts`
- Mock transcribeFn
- First temp accepts → no retry
- All temps reject → returns last
- No_speech at any temp → immediate return

---

### Phase 4: Condition-on-Previous-Text (TDD)

**File:** `src/models/whisper-seq2seq/chunk-context.ts`

```ts
export interface ChunkContextOptions {
  maxContextTokens: number;    // default: maxTargetPositions / 2
  resetOnFallback: boolean;    // default: true
}

export class ChunkContextBuilder {
  private allTokens: number[] = [];
  private resetSince: number = 0;

  getPreviousTokens(): number[];
  addSegmentTokens(tokens: number[]): void;
  reset(): void;
  getTotalTokenCount(): number;
}

// Build prompt with previous context
// [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>,
//  <|0.00|>, ...previousTokens..., <|timestamp|>]
export function buildPromptWithContext(
  basePrompt: readonly number[],
  previousTokens: readonly number[],
  maxContextTokens: number,
): number[];
```

**Test file:** `tests/whisper-chunk-context.test.ts`

---

### Phase 5: Drift Handler (TDD)

**File:** `src/models/whisper-seq2seq/drift-handler.ts`

```ts
export interface DriftCorrectionResult {
  start: number;
  end: number;
  corrected: boolean;
}

export class DriftHandler {
  private seekSamples: number = 0;

  reset(audioLengthSamples: number): void;
  getSeekSeconds(sampleRate: number): number;
  advanceBy(durationSeconds: number, sampleRate: number): void;
  correctTimestamps(
    modelStartSec: number,
    modelEndSec: number,
    sampleRate: number,
    maxDriftSec?: number,  // default 1.0
  ): DriftCorrectionResult;
}
```

Algorithm (from whisper.cpp):
- External seek counter tracks absolute position
- After each segment, advance by model's predicted duration
- If model timestamps diverge > 1s from seek, use seek instead
- Prevents cumulative drift in long recordings

**Test file:** `tests/whisper-drift-handler.test.ts`

---

### Phase 6: VAD Segmenter (TDD)

**File:** `src/models/whisper-seq2seq/vad-segmenter.ts`

This is where we use the project's existing VAD backends.

```ts
import type { VadSegmenterConfig } from './enhanced-types.js';

export interface VadSpeechSegment {
  startSeconds: number;
  endSeconds: number;
  durationSeconds: number;
}

// Segment audio using TenVAD or FireRed VAD
// Returns speech segments suitable for Whisper 30s windows
export async function segmentAudioWithVad(
  audioData: Float32Array,
  sampleRate: number,
  config: VadSegmenterConfig,
): Promise<VadSpeechSegment[]>;

// Merge segments that are close together
export function mergeVadSegments(
  segments: VadSpeechSegment[],
  minSilenceDurationMs: number,
  speechPadMs: number,
  maxSegmentDurationMs: number,
): VadSpeechSegment[];
```

**VAD backend selection:**

| Backend | When to use | How it works |
|---------|-------------|--------------|
| `ten-vad` | Browser + Node, fast, WASM-based | `TenVadAdapter` from `src/runtime/ten-vad-browser.ts` |
| `firered-vad` | Browser + Node, more features (AED, streaming) | `FireRedVad` from `src/runtime/firered-vad/api/classes.ts` |

Both backends:
- Return speech probability per frame
- Already have browser worker wrappers
- Already integrated into `StreamingDetector`
- Already used in `vad-demo` and `streaming-demo`

The VAD segmenter bridges the existing runtime VAD to the Whisper model family:
1. Run VAD on full audio → get probability timeline
2. Threshold to get speech/non-speech segments
3. Merge short segments, pad edges
4. Cap at 29s (under Whisper's 30s window)
5. Return segments for Whisper to process independently

**Test file:** `tests/whisper-vad-segmenter.test.ts`
- Mock VAD probabilities → expected segments
- Merge: close segments merged, far segments kept
- Padding: segments padded but clamped to [0, duration]
- Max duration: long speech split at 29s

---

### Phase 7: Segment Merger (TDD)

**File:** `src/models/whisper-seq2seq/segment-merger.ts`

```ts
import type { WhisperNativeSegment, WhisperNativeWord } from './types.js';

export interface MergedSegments {
  segments: WhisperNativeSegment[];
  words: WhisperNativeWord[];
}

// After Whisper processes each VAD segment independently:
// 1. Adjust timestamps from chunk-relative to absolute
// 2. Merge overlapping segments
// 3. Deduplicate words at boundaries
export function mergeWhisperSegments(
  perChunkResults: Array<{
    segments: WhisperNativeSegment[];
    words: WhisperNativeWord[];
    timeOffsetSeconds: number;
  }>,
): MergedSegments;
```

**Test file:** `tests/whisper-segment-merger.test.ts`

---

### Phase 8: Enhanced Executor (TDD)

**File:** `src/models/whisper-seq2seq/enhanced-executor.ts`

```ts
import type { WhisperExecutor, WhisperDecodeContext } from './types.js';
import type { WhisperSeq2SeqTranscriptionOptions } from './types.js';
import type { EnhancedDecodeOptions, EnhancedTranscript } from './enhanced-types.js';

export class EnhancedWhisperExecutor implements WhisperExecutor {
  constructor(
    private readonly vanilla: WhisperExecutor,
    private readonly vadSegmenter?: VadSegmenterConfig,
  );

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions & EnhancedDecodeOptions,
    context: WhisperDecodeContext,
  ): Promise<EnhancedTranscript>;

  dispose(): Promise<void>;
}
```

Composition pattern:
```
EnhancedWhisperExecutor
  ├── WhisperOnnxExecutor (vanilla — all ONNX work)
  ├── QualityGates (compression, logprob, entropy, no_speech)
  ├── TemperatureFallback (retry with escalating temperatures)
  ├── ChunkContextBuilder (condition-on-previous-text)
  ├── DriftHandler (seek counter for long audio)
  ├── VadSegmenter (TenVAD or FireRed VAD)
  └── SegmentMerger (overlap reconciliation)
```

Pipeline:
1. If long audio + VAD enabled: pre-segment with VAD
2. For each chunk:
   a. Build prompt (with previous context if condition_on_previous_text)
   b. Call vanilla.transcribe() at temperature 0.0
   c. Run quality gates
   d. If rejected, retry with temperature fallback
   e. If no_speech, skip segment
   f. Correct timestamps with drift handler
   g. Add segment tokens to context
3. Merge all segments
4. Return EnhancedTranscript with quality metrics

**Test file:** `tests/whisper-enhanced-executor.test.ts`
- Integration test with mock vanilla executor
- Quality gate rejection triggers fallback
- Context conditioning carries between segments
- Drift correction applies to timestamps
- VAD segmentation produces correct chunks

---

### Phase 9: Batched Encoder (FUTURE)

**File:** `src/models/whisper-seq2seq/batched-encoder.ts` (NOT YET)

WhisperX's key optimization: batch multiple audio segments into a single encoder forward pass.

```ts
// Future: batched encoder for GPU
export async function batchedEncode(
  encoderSession: Any,
  melFeatures: Float32Array[],  // multiple mel spectrograms
): Promise<Float32Array[]>;
```

Only beneficial on GPU with multiple concurrent requests. Premature for first release.

---

### Phase 10: WAV2VEC2 Alignment (FUTURE)

**File:** `src/models/whisper-seq2seq/wav2vec2-aligner.ts` (NOT YET)

WhisperX's most significant innovation: frame-level word alignment using WAV2VEC2 forced alignment.

Requirements:
- Separate ONNX model per language
- CTC forced alignment (Viterbi algorithm)
- ~20ms resolution vs Whisper DTW's ~100ms

This is a separate model loading + alignment pipeline. Can be added as an optional post-processor.

---

### Phase 11: Diarization (FUTURE)

**File:** `src/models/whisper-seq2seq/diarize.ts` (NOT YET)

Speaker diarization via pyannote or similar.

Requirements:
- Gated model from HuggingFace (needs HF token)
- Word-level speaker assignment
- Speaker segment grouping

This is a completely separate pipeline that can be layered on top of enhanced executor output.

---

## Dependency Graph

```
Phase 1 (enhanced-types.ts)
  ↓
Phase 2 (quality-gates.ts) ← depends on types
  ↓
Phase 3 (temperature-fallback.ts) ← depends on types + gates
  ↓
Phase 4 (chunk-context.ts) ← independent
  ↓
Phase 5 (drift-handler.ts) ← independent
  ↓
Phase 6 (vad-segmenter.ts) ← depends on types + existing runtime VAD
  ↓
Phase 7 (segment-merger.ts) ← depends on types
  ↓
Phase 8 (enhanced-executor.ts) ← depends on ALL above

Phase 9-11 (FUTURE, not blocking)
```

Phases 2, 4, 5, 6, 7 can be developed in parallel by different agents.

## What NOT to Do

1. Do NOT touch `core.ts`, `executor.ts`, `processors.ts` — other agent's domain
2. Do NOT start WebGPU work
3. Do NOT implement mixed dtype or q4/q4f16
4. Do NOT import from `executor.ts` in new modules — depend only on types + core
5. Do NOT add Silero VAD — use TenVAD or FireRed VAD instead
6. Do NOT create framework-specific code in the core package

## TDD Discipline

For each phase:
1. Write test file FIRST (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor if needed (REFACTOR)
4. Run gate: `npm run typecheck && npm run lint && npm test && npm run build`

## VAD Integration Details

### Why TenVAD and FireRed instead of Silero

| Feature | Silero | TenVAD | FireRed VAD |
|---------|--------|--------|-------------|
| Already in project | No | Yes | Yes |
| Browser WASM | Needs download | Bundled (`ten_vad.wasm`) | ONNX Runtime |
| Streaming support | External | Yes (worker) | Yes (worker + stream) |
| AED (Acoustic Event Detection) | No | No | Yes |
| File-mode batch processing | No | No | Yes |
| Browser demo | No | vad-demo | firered-vad-web |
| Dependency | torch or ONNX | None (WASM) | ONNX Runtime |

The VAD segmenter for Whisper chunking needs:
1. Run VAD on full audio → probability timeline
2. Threshold to get speech segments
3. Merge + pad + cap at 29s

Both TenVAD and FireRed can do this. The `VadSegmenterConfig.backend` field lets the user choose.

### How to integrate with existing VAD

The `VoiceActivityProbabilityBuffer` in `src/runtime/vad.ts` already maintains a probability timeline. For Whisper chunking, we need a simpler interface:

```ts
// New interface in vad-segmenter.ts:
interface WhisperVadBackend {
  segment(audio: Float32Array, sampleRate: number, threshold: number): Promise<VadSpeechSegment[]>;
}

// Adapter for TenVAD:
class TenVadSegmenter implements WhisperVadBackend { ... }

// Adapter for FireRed:
class FireRedVadSegmenter implements WhisperVadBackend { ... }
```

This decouples the Whisper model family from the runtime VAD implementation.

## Acceptance Criteria

1. All new files under `src/models/whisper-seq2seq/`
2. No changes to existing files (core.ts, executor.ts, processors.ts, etc.)
3. Quality gates are pure functions, independently testable
4. Temperature fallback is a retry loop (no ONNX changes)
5. VAD segmenter uses TenVAD or FireRed (not Silero)
6. EnhancedWhisperExecutor wraps WhisperExecutor via composition
7. All 4 quality gates pass unit tests
8. VAD segmenter passes unit tests with mock VAD
9. Enhanced executor passes integration test with mock vanilla
10. Gate passes: typecheck, lint, tests, build
