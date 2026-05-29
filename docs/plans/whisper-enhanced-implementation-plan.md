# Whisper Vanilla + Enhanced: Concrete Implementation Plan

## Current State (branch: feat/asr-pipeline-output-formats)

### Already Done (by other agent)
- `core.ts` — WhisperCoreSession interface + whisperGreedyDecode (149 lines)
- `processors.ts` — WhisperTimestampLogitProcessor (104 lines)
- `executor.ts` — WhisperOnnxExecutor class (1389 lines, monolithic)
  - Merged decoder + splitgraph decoder paths
  - Alignment via decoder_align.onnx + DTW
  - KV cache management
  - Prompt construction
  - Session creation + artifact resolution
  - Long audio chunking (basic 30s window)
- `tokenizer.ts` — Whisper BPE tokenizer
- `ort.ts` — ONNX session creation + artifact resolution
- `types.ts` — interfaces + types
- `beam-search.ts` — beam search skeleton (2241 bytes)
- `chunking.ts` — chunking skeleton (2241 bytes)
- `attention-alignment.ts` — DTW alignment (7276 bytes)
- `word-timestamps.ts` — word timestamp extraction
- `generation-config.ts` — generation config parsing
- 27 test files for whisper components
- 2 smoke validators (V1 + V2)

### Architecture Already Separated
The other agent has already started extracting `core.ts` as a pure decode loop.
This is exactly the "vanilla" foundation we need.

## Implementation Plan

### Phase 0: Refactor executor.ts → clean vanilla (PROTECTED — other agent's domain)

**WARNING:** executor.ts is actively being worked on by another agent.
DO NOT TOUCH executor.ts directly. Instead, create new files that compose with it.

The other agent is responsible for:
- Continuing the core.ts extraction from executor.ts
- Cleaning up the monolithic executor
- Making WhisperOnnxExecutor compose with core.ts

Our work builds ON TOP of what they extract. We create new files only.

---

### Phase 1: Enhanced Types & Interfaces

**File: `src/models/whisper-seq2seq/enhanced-types.ts`** (NEW)

```ts
// Quality gate result
type QualityVerdict = 'accept' | 'reject' | 'no_speech';

interface QualityGateResult {
  verdict: QualityVerdict;
  compressionRatio?: number;
  avgLogProb?: number;
  noSpeechProb?: number;
  entropy?: number;
  reason?: string;
}

// Enhanced decode options (extends vanilla)
interface EnhancedDecodeOptions {
  // Quality thresholds
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

  // Grammar constraint
  grammar?: GrammarSpec;
}

interface GrammarSpec {
  type: 'choice' | 'regex' | 'cfg';
  rules: Record<string, string>;
}

// Enhanced transcript includes quality metrics
interface EnhancedTranscript extends WhisperNativeTranscript {
  qualityMetrics?: {
    compressionRatio: number;
    avgLogProb: number;
    noSpeechProb: number;
    entropy: number;
    temperature: number;
  };
}
```

**Tests:** `tests/whisper-enhanced-types.test.ts`
- Type-level tests (compile-time validation)
- QualityGateResult factory tests

---

### Phase 2: Quality Gates

**File: `src/models/whisper-seq2seq/quality-gates.ts`** (NEW)

Pure functions, no ONNX dependency. Each gate is a callback:

```ts
type QualityGate = (result: DecodeResultWithMetrics) => QualityGateResult;

// Gates to implement:
function compressionRatioGate(threshold: number): QualityGate;
function logProbGate(threshold: number): QualityGate;
function noSpeechGate(threshold: number, logProbThreshold: number): QualityGate;
function entropyGate(threshold: number): QualityGate;

// Composite gate runner
function evaluateGates(
  result: DecodeResultWithMetrics,
  gates: QualityGate[],
): QualityGateResult;
```

#### Sub-task 2a: Compression Ratio (zlib-equivalent in JS)

```ts
// Use CompressionStream API (available in Node 18+ and modern browsers)
// or pako (already in transformers.js dependency tree)
//
// Algorithm: len(text_bytes) / len(compressed(text_bytes))
// Threshold: 2.4 (same as OpenAI/faster-whisper/whisper.cpp)
// High ratio = repetitive text = hallucination
```

**Decision needed:** Use pako or implement zlib-equivalent?
- pako is 45KB, well-tested, already used by transformers.js
- CompressionStream is streaming, different API
- Recommendation: pako (already in dependency tree)

#### Sub-task 2b: Log Probability Collection

```ts
// Collect per-token log probabilities during decode loop
// Requires extending WhisperCoreSession to return logProbs alongside logits
// OR: compute from logits directly: logProb = logits[token] - log(sum(exp(logits)))
//
// Key: avg_logprob = sum(logProbs) / numGeneratedTokens
// Threshold: -1.0
```

**Decision needed:** Extend core.ts session interface or compute from logits?
- Recommendation: Compute from logits (no interface change needed)
- `logProb = logits[chosenToken] - logSumExp(logits)`
- This matches what CTranslate2 does internally

#### Sub-task 2c: No-Speech Probability

```ts
// Extract no_speech_token probability from first generated token logits
// Token 50362 (<|nospeech|>)
// Dual check: noSpeechProb > 0.6 AND avgLogProb < -1.0
// Both must be true → segment is silence
```

#### Sub-task 2d: Entropy Calculation

```ts
// Compute Shannon entropy of the logit distribution
// H = -sum(p * log(p)) where p = softmax(logits)
// Threshold: 2.4 nats (same as whisper.cpp)
// High entropy = model uncertain = hallucination
```

**Tests:** `tests/whisper-quality-gates.test.ts`
- Unit tests for each gate with known inputs
- Composite gate evaluation tests
- Edge cases: empty text, single token, perfect confidence

---

### Phase 3: Temperature Fallback

**File: `src/models/whisper-seq2seq/temperature-fallback.ts`** (NEW)

```ts
interface TemperatureFallbackOptions {
  temperatures: number[];           // default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  gates: QualityGate[];
}

async function transcribeWithFallback(
  transcribeFn: (temperature: number) => Promise<DecodeResultWithMetrics>,
  options: TemperatureFallbackOptions,
): Promise<{ result: DecodeResultWithMetrics; temperature: number; attempts: number }>;
```

Algorithm:
1. Call transcribeFn(0.0) → get result
2. Run quality gates → if accept, return
3. If no_speech, return empty result
4. If reject, try next temperature
5. After all temperatures exhausted, return last result

**Tests:** `tests/whisper-temperature-fallback.test.ts`
- Mock transcribeFn to test fallback sequence
- Test: first temp accepts → no retry
- Test: all temps reject → returns last
- Test: no_speech verdict → immediate return with empty

---

### Phase 4: Condition-on-Previous-Text

**File: `src/models/whisper-seq2seq/chunk-context.ts`** (NEW)

```ts
interface ChunkContextOptions {
  maxContextTokens: number;  // default: maxTargetPositions / 2
  resetOnFallback: boolean;  // default: true
}

class ChunkContextBuilder {
  private allTokens: number[] = [];
  private resetSince: number = 0;

  // Get previous tokens for next chunk's prompt
  getPreviousTokens(): number[];

  // Add a segment's tokens to context
  addSegmentTokens(tokens: number[]): void;

  // Reset context (called when segment triggers fallback/hallucination)
  reset(): void;
}
```

Integration with prompt construction:
- Whisper prompt: [SOT, lang, task, nots, <timestamp>, ...previousTokens..., <timestamp>]
- Cap previousTokens to maxContextTokens
- Reset if previous segment was hallucinated

**Tests:** `tests/whisper-chunk-context.test.ts`
- Context accumulation
- Token cap enforcement
- Reset behavior
- Integration with prompt construction

---

### Phase 5: Drift Handler

**File: `src/models/whisper-seq2seq/drift-handler.ts`** (NEW)

```ts
class DriftHandler {
  private seekPosition: number = 0;  // in samples

  // Initialize seek position
  reset(audioLength: number): void;

  // Get expected start time for current chunk
  getExpectedStart(sampleRate: number): number;

  // Advance seek by segment duration (in samples)
  advanceBy(durationSamples: number): void;

  // Correct timestamps if model drifts too far from seek
  correctTimestamps(
    modelStart: number,
    modelEnd: number,
    sampleRate: number,
  ): { start: number; end: number };

  // Get current seek position
  getSeek(): number;
}
```

Algorithm (from whisper.cpp):
- Maintain external seek counter (sample-level)
- After each segment, advance seek by model's predicted duration
- If model timestamps diverge > 1s from seek, use seek instead
- This prevents cumulative drift in long recordings

**Tests:** `tests/whisper-drift-handler.test.ts`
- Seek advancement
- Drift correction threshold
- Edge: short segments, long segments
- Edge: model predicts backwards timestamps

---

### Phase 6: Enhanced Executor

**File: `src/models/whisper-seq2seq/enhanced-executor.ts`** (NEW)

```ts
class EnhancedWhisperExecutor implements WhisperExecutor {
  private vanilla: WhisperOnnxExecutor;
  private vadSession?: any;  // Silero VAD ONNX session

  constructor(vanilla: WhisperOnnxExecutor);

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions & EnhancedDecodeOptions,
    context: WhisperDecodeContext,
  ): Promise<EnhancedTranscript>;
}
```

Composition pattern:
```
EnhancedWhisperExecutor
  ├── WhisperOnnxExecutor (vanilla — does all ONNX work)
  ├── QualityGates (compression, logprob, entropy, no_speech)
  ├── TemperatureFallback (retry with escalating temperatures)
  ├── ChunkContextBuilder (condition-on-previous-text)
  ├── DriftHandler (seek counter for long audio)
  └── (future) VadChunker, GrammarConstraint
```

The enhanced executor:
1. If long audio: use VAD chunking OR 30s windows
2. For each chunk:
   a. Build prompt (with previous context if condition_on_previous_text)
   b. Call vanilla.transcribe() at temperature 0.0
   c. Run quality gates
   d. If rejected, retry with temperature fallback
   e. If no_speech, skip segment
   f. Correct timestamps with drift handler
   g. Add segment tokens to context
3. Merge all segments into final transcript

**Tests:** `tests/whisper-enhanced-executor.test.ts`
- Integration test with mock vanilla executor
- Test: quality gate rejection triggers fallback
- Test: context conditioning carries between segments
- Test: drift correction applies to timestamps

---

### Phase 7: Grammar Constraints (DEFERRED)

Not blocking. Documented for later.
- Logit masking based on CFG
- Same approach as whisper.cpp / llama.cpp
- Only needs logit processor, no model changes

### Phase 8: VAD Integration (DEFERRED)

Not blocking. Documented for later.
- Silero VAD as separate ONNX model
- Pre-segmentation before Whisper
- Segment merging + padding

### Phase 9: WASM Comparison Runner (AFTER Phase 6)

**File: `tests/smoke/whisper-enhanced-wasm-compare.mjs`** (NEW)

Follow the V2 pattern:
- Session reuse across variants
- Quality gate evaluation per segment
- Temperature fallback logging
- Fair language comparison (fix the Turkish issue)

---

## Dependency Graph

```
Phase 1 (types) ─────────────────────────────────────────┐
                                                          │
Phase 2 (quality gates) ─── depends on Phase 1 ──────────┤
  2a: compression ratio (needs pako)                      │
  2b: log probability (compute from logits)               │
  2c: no-speech prob (extract from logits)                │
  2d: entropy (compute from logits)                       │
                                                          │
Phase 3 (temperature fallback) ── depends on Phase 2 ────┤
                                                          │
Phase 4 (chunk context) ── independent ───────────────────┤
                                                          │
Phase 5 (drift handler) ── independent ───────────────────┤
                                                          │
Phase 6 (enhanced executor) ── depends on 1,2,3,4,5 ─────┤
                                                          │
Phase 7 (grammar) ── DEFERRED                             │
Phase 8 (VAD) ── DEFERRED                                 │
Phase 9 (WASM runner) ── depends on Phase 6 ─────────────┘
```

## What NOT to Do (from user's prompt)

1. Do NOT touch executor.ts — other agent's domain
2. Do NOT start WebGPU work
3. Do NOT implement mixed dtype
4. Do NOT implement q4/q4f16
5. Do NOT start any work that conflicts with the active agent

## TDD Approach

For each phase:
1. Write tests FIRST (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor if needed (REFACTOR)
4. Run full gate: typecheck + lint + test + build

## Acceptance Criteria

1. All new files under `src/models/whisper-seq2seq/`
2. No changes to existing files (executor.ts, core.ts, processors.ts)
3. EnhancedWhisperExecutor wraps WhisperOnnxExecutor via composition
4. Quality gates are pure functions, independently testable
5. Temperature fallback is a retry loop around vanilla transcribe
6. All 4 gates pass unit tests
7. Enhanced executor passes integration test with mock vanilla
8. Gate passes: typecheck, lint, 84+ tests, build
