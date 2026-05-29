# Whisper Enhanced Executor: Agent Handoff

## Who should read this

The agent currently working on `executor.ts` and `core.ts` in `src/models/whisper-seq2seq/`.

## What we did (Flexo, session 2026-05-29)

1. **Cataloged 18 production features** from faster-whisper, WhisperX, whisper.cpp
2. **Designed the vanilla + enhanced architecture** — split at the decode orchestration layer
3. **Wrote the full implementation plan** — 11 phases, TDD, file-by-file

## Architecture summary

```
Layer 0: core.ts (YOU — DONE)
  Pure decode loop, ONNX-agnostic, backend-agnostic

Layer 1: executor.ts (YOU — DONE)
  ONNX bridge, artifact resolution, session management

Layer 2: Enhanced decode features (US — NEW FILES)
  quality-gates.ts, temperature-fallback.ts, chunk-context.ts, drift-handler.ts

Layer 3: Smart chunking (US — uses existing VAD)
  vad-segmenter.ts, segment-merger.ts

Layer 4: Advanced (FUTURE)
  batched-encoder.ts, wav2vec2-aligner.ts, diarize.ts
```

## Key constraint: composition, not modification

We will NOT modify your files. We create new files that compose with `WhisperOnnxExecutor`:

```ts
// Our enhanced-executor.ts wraps your executor via composition
class EnhancedWhisperExecutor implements WhisperExecutor {
  constructor(private readonly vanilla: WhisperExecutor) { ... }
  async transcribe(audio, options, context) {
    // 1. VAD pre-segmentation (if enabled)
    // 2. Per-chunk: call this.vanilla.transcribe()
    // 3. Quality gates on each result
    // 4. Temperature fallback if quality fails
    // 5. Condition-on-previous-text between chunks
    // 6. Drift correction on timestamps
    // 7. Merge segments
  }
}
```

## What we need from you

1. **Stable `WhisperExecutor` interface** — we depend on `transcribe()` returning `WhisperNativeTranscript`
2. **Stable `WhisperCoreSession` interface** — quality gates need access to logits
3. **Logit collection** — our quality gates need per-token logits. Currently `whisperGreedyDecode` doesn't collect them. Consider adding an optional logit callback or returning logits in `WhisperDecodeResult`:

```ts
// Option A: callback (minimal change)
interface WhisperDecodeOptions {
  // ... existing ...
  readonly onTokenLogits?: (tokenIndex: number, logits: Float32Array, chosenToken: number) => void;
}

// Option B: extended result
interface WhisperDecodeResult {
  readonly tokens: readonly number[];
  readonly tokenLogits?: readonly Float32Array[];  // per-token logits (optional)
}
```

Option A is cleaner — no memory overhead unless someone needs it.

4. **No-speech token probability** — `noSpeechProb` is the probability of token 50362 at the first generated position. If you can expose this (or first-token logits), we can compute it in the quality gate.

## VAD decision

We will use **TenVAD** and **FireRed VAD** (already in `src/runtime/`), NOT Silero VAD.

Reasons:
- Both already implemented and tested in the project
- TenVAD: WASM-based, fast, bundled, browser+Node
- FireRed VAD: ONNX-based, streaming, file-mode, AED support
- Both already integrated into `StreamingDetector` and used in demos
- No new dependencies needed

The VAD segmenter will have a simple adapter interface:

```ts
interface WhisperVadBackend {
  segment(audio: Float32Array, sampleRate: number, threshold: number): Promise<VadSpeechSegment[]>;
}
```

## Files we will create (no conflicts with you)

```
src/models/whisper-seq2seq/
  enhanced-types.ts       — types for quality gates, fallback, metrics
  quality-gates.ts        — compression ratio, logprob, entropy, no-speech
  temperature-fallback.ts — temperature schedule + retry loop
  chunk-context.ts        — condition-on-previous-text prompt builder
  drift-handler.ts        — seek counter + timestamp drift correction
  vad-segmenter.ts        — VAD-based audio pre-segmentation
  segment-merger.ts       — overlap reconciliation + timestamp adjustment
  enhanced-executor.ts    — wraps WhisperExecutor + all enhanced features
```

## Files we will NOT touch

```
src/models/whisper-seq2seq/
  core.ts           ← yours
  executor.ts       ← yours
  processors.ts     ← yours
  tokenizer.ts      ← yours
  ort.ts            ← yours
  types.ts          ← yours (we may add types but won't modify existing)
  beam-search.ts    ← yours
  chunking.ts       ← yours
  attention-alignment.ts ← yours
  word-timestamps.ts     ← yours
  generation-config.ts   ← yours
  manifest.ts            ← yours
  mapping.ts             ← yours
  local-file.ts          ← yours
  config.ts              ← yours
  model.ts               ← yours
  index.ts               ← yours
```

## Reference documents

All in `docs/plans/`:
- `whisper-vanilla-enhanced-architecture.md` — architecture overview, feature catalog, API design
- `whisper-enhanced-implementation-plan.md` — 11-phase plan, TDD, file structure, acceptance criteria

All in `docs/references/`:
- `whisper-reference-decode-patterns.md` — comparative study of 4 implementations
- `whisper-webgpu-smoke-notes.md` — WebGPU fp16 smoke guide

## Coordination

- Branch: `feat/asr-pipeline-output-formats` (shared)
- Do not push uncommitted changes to files we might import from
- If you change `WhisperExecutor` interface, let us know
- If you add logit collection to `core.ts`, that unlocks our Phase 2

## TL;DR

You build the engine (core.ts + executor.ts). We build the turbocharger (quality gates, temperature fallback, VAD chunking, drift correction). Both share the same ONNX graphs. No file conflicts.
