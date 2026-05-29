# Whisper Vanilla + Enhanced Architecture Plan

## Concept

```
VanillaWhisperExecutor          EnhancedWhisperExecutor
    (reference-accurate)    ←──      (best-of-breed)
         |                              |
    Same 4 ONNX graphs           Wraps vanilla + adds:
    Same prompt tokens           - Temperature fallback
    Same decode loop             - Quality gates (compression, logprob, entropy)
    Same logit processors        - VAD chunking (Silero)
    Same KV cache mgmt           - Condition-on-previous-text
    Same beam search             - Drift handling
    Same alignment (DTW)         - Batched encoder
                                 - Grammar constraints
                                 - WAV2VEC2 alignment (optional)
                                 - Diarization (optional)
```

Both share the same ONNX graphs, same mel frontend, same tokenizer, same artifact resolution.
The split is purely in the decode orchestration layer.

## Why This Split Works

1. **All enhanced features are runtime-only** — no ONNX graph changes needed
2. **Vanilla is testable against OpenAI/HF reference** — byte-exact parity
3. **Enhanced builds on proven vanilla** — not a parallel implementation
4. **Each enhanced feature is independent** — can be added one at a time
5. **Vanilla stays clean** — no feature creep, always the fallback

## File Structure

```
src/models/whisper-seq2seq/
  # Core (shared by vanilla + enhanced)
  types.ts              — shared types, interfaces
  config.ts             — model classification
  ort.ts                — ONNX session creation + artifact resolution
  tokenizer.ts          — Whisper BPE tokenizer
  processors.ts         — logit processors (suppress, timestamps)
  generation-config.ts  — generation config parsing
  manifest.ts           — manifest parsing
  mapping.ts            — preset mapping
  local-file.ts         — local file loading
  attention-alignment.ts — DTW alignment shared code
  word-timestamps.ts    — word timestamp extraction
  index.ts              — barrel exports

  # Vanilla executor (reference-accurate)
  executor.ts           — VanillaWhisperExecutor + splitGraphDecodeLoop
    - Greedy decode (AR loop with EOS/max_tokens)
    - Beam search (if numBeams > 1)
    - KV cache management (present → past_key_values)
    - Logit processing (suppress + timestamp rules)
    - Alignment via decoder_align.onnx + DTW
    - Prompt construction (SOT + lang + task + nots)
    - Language detection (encoder + SOT-only decode)
    - Both merged and splitgraph paths
    - 30s fixed-window chunking (simple)

  # Enhanced executor (wraps vanilla)
  enhanced-executor.ts  — EnhancedWhisperExecutor
    - Wraps VanillaWhisperExecutor
    - Adds quality gates after each segment
    - Adds temperature fallback
    - Adds VAD-based chunking
    - Adds condition-on-previous-text
    - Adds drift handling
    - Adds entropy filter
    - Adds grammar constraints

  # Enhanced subsystems
  quality-gates.ts      — compression ratio, logprob, entropy, no-speech
  temperature-fallback.ts — temperature schedule + retry logic
  vad-chunking.ts       — Silero VAD integration for pre-segmentation
  drift-handler.ts      — seek counter + drift correction
  grammar.ts            — CFG grammar-constrained decoding
  chunk-context.ts      — condition-on-previous-text prompt builder
```

## Vanilla API

```ts
// The vanilla executor is the foundation. It does ONE thing correctly:
// Given mel features + options → tokens + alignment + timestamps

interface VanillaTranscribeOptions {
  language: string;                    // 'en', 'tr', 'auto'
  task: 'transcribe' | 'translate';
  noTimestamps: boolean;
  maxNewTokens: number;
  numBeams: number;                    // 1 = greedy, >1 = beam search
  temperature: number;                 // 0 = greedy argmax
  suppressTokens: number[];
  beginSuppressTokens: number[];
  alignmentHeads: Array<{layer: number, head: number}>;
}

class VanillaWhisperExecutor {
  // Core: encode → decode → align
  async transcribe(audio, options): WhisperTranscript;
  
  // Low-level: expose for enhanced wrapper
  async encode(mel): encoderHiddenStates;
  async decodeGreedy(encoderOutput, prompt, options): DecodeResult;
  async decodeBeam(encoderOutput, prompt, options): DecodeResult[];
  async align(tokens, encoderOutput): AlignmentResult;
  async detectLanguage(mel): string;
}
```

## Enhanced API

```ts
// The enhanced executor wraps vanilla and adds production features
interface EnhancedTranscribeOptions extends VanillaTranscribeOptions {
  // Quality gates
  compressionRatioThreshold?: number;  // default 2.4
  logProbThreshold?: number;           // default -1.0
  entropyThreshold?: number;           // default 2.4
  noSpeechThreshold?: number;          // default 0.6
  
  // Temperature fallback
  temperatureFallback?: boolean;       // default true
  temperatures?: number[];             // default [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  
  // Chunking
  vadChunking?: boolean;               // default false
  vadOptions?: VadOptions;
  conditionOnPreviousText?: boolean;   // default true
  maxContextTokens?: number;           // default max_length // 2
  
  // Grammar
  grammar?: GrammarSpec;
}

class EnhancedWhisperExecutor {
  constructor(vanilla: VanillaWhisperExecutor);
  async transcribe(audio, options): EnhancedTranscript;
}
```

## Implementation Order

### Phase 1: Vanilla (current — already done)

- [x] Greedy decode with AR loop
- [x] KV cache management (splitgraph)
- [x] Logit suppression (suppress + timestamp rules)
- [x] Alignment via decoder_align.onnx + DTW
- [x] Prompt construction
- [x] fp32/fp16 exact parity
- [x] Session reuse
- [ ] Beam search
- [ ] Language detection
- [ ] 30s fixed-window chunking
- [ ] Clean up executor.ts to be the "vanilla" reference

### Phase 2: Quality Gates

- [ ] Compression ratio calculation (zlib-equivalent in JS)
- [ ] Log probability collection during decode
- [ ] No-speech probability extraction from logits
- [ ] Entropy calculation
- [ ] Temperature fallback loop
- [ ] These are all pure runtime — no ONNX changes

### Phase 3: Smart Chunking

- [ ] Silero VAD integration (separate ONNX model)
- [ ] VAD segment merging + padding
- [ ] Condition-on-previous-text prompt construction
- [ ] Drift handling via seek counter
- [ ] Segment overlap reconciliation

### Phase 4: Advanced (optional, later)

- [ ] Batched encoder (multiple segments in parallel)
- [ ] Grammar-constrained decoding (logit masking)
- [ ] WAV2VEC2 alignment (separate model per language)
- [ ] Diarization (separate pipeline)

## Key Design Decisions

### 1. Wrap, don't inherit

EnhancedWhisperExecutor wraps VanillaWhisperExecutor via composition.
This keeps vanilla clean and testable. Enhanced delegates core work to vanilla
and adds pre/post-processing around it.

### 2. Quality gates are per-segment callbacks

Instead of embedding quality checks in the decode loop, they are callbacks:

```ts
type SegmentQualityGate = (result: DecodeResult) => 'accept' | 'reject' | 'no_speech';

const defaultGates: SegmentQualityGate[] = [
  compressionRatioGate(2.4),
  logProbGate(-1.0),
  noSpeechGate(0.6, -1.0),
  entropyGate(2.4),
];
```

This makes quality gates composable, testable, and optional.

### 3. Temperature fallback is a retry loop

```ts
async function transcribeWithFallback(
  vanilla: VanillaWhisperExecutor,
  audio: Audio,
  options: EnhancedTranscribeOptions,
): Promise<WhisperTranscript> {
  const temperatures = options.temperatures ?? [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
  let lastResult = null;
  
  for (const temp of temperatures) {
    const result = await vanilla.transcribe(audio, { ...options, temperature: temp });
    
    const verdict = evaluateGates(result, options);
    if (verdict === 'accept') return result;
    if (verdict === 'no_speech') return { ...result, text: '' };
    
    lastResult = result;
  }
  
  return lastResult; // Return last attempt if all temperatures failed
}
```

### 4. VAD is a pre-processing step, not embedded

VAD runs on the full audio BEFORE any Whisper inference:
1. Load Silero VAD ONNX model
2. Run on full audio → speech segments
3. Feed each segment to vanilla.transcribe()
4. Merge results with timestamp adjustment

This means VAD is completely optional and can be swapped (Silero, pyannote, etc.)

### 5. Vanilla is always the fallback

If enhanced features cause issues, users can always drop down to vanilla:
- Same API shape
- Same ONNX graphs
- Same output format
- Just fewer quality gates / chunking

## Complexity Comparison: What We're Building vs What Exists

```
                              | Lines of decode logic | Features
------------------------------+-----------------------+----------
OpenAI Whisper (Python)       | ~600                  | 10 vanilla
faster-whisper (Python)       | ~1200                 | 10 vanilla + 6 quality + VAD
WhisperX (Python)             | ~2000                 | 10 vanilla + VAD + WAV2VEC2 + diarize
whisper.cpp (C++)             | ~3000                 | 10 vanilla + quality + grammar + drift
                              |                       |
asrjs Vanilla (TS) target     | ~800                  | 10 vanilla (match OpenAI)
asrjs Enhanced (TS) target    | ~400                  | quality + chunking + grammar
asrjs total                   | ~1200                 | Best of all three
```

## What NOT to Include

1. **WAV2VEC2 alignment** — requires a separate model per language. Use Whisper's native DTW instead.
2. **Diarization** — requires pyannote gated model + HF token. Out of scope for core library.
3. **Batched encoder** — only beneficial on GPU with multiple concurrent requests. Premature for first release.
4. **AOT/mmap loading** — GGML-specific, doesn't apply to ONNX runtime.
5. **Speculative decoding** — requires draft model, complex, not worth it yet.
