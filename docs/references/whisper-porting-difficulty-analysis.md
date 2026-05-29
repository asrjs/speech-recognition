# Whisper Porting Difficulty Analysis

Why Whisper is 4-13x harder to port than NeMo TDT or MedASR, and how to approach it.

---

## TL;DR

NeMo TDT and MedASR were "single-pass encoder" models — no decoder loop, no KV cache,
no autoregressive state. Whisper is a full seq2seq model with 4 ONNX graphs, KV cache
state machine, dual decoder paths, logit processing pipeline, and 30s chunking.
The other agents aren't struggling because they're bad — they're struggling because
Whisper has 13 complexity points vs 3 for NeMo TDT and 1 for MedASR.

---

## Architecture Comparison

```
Model          | ONNX  | Decoder      | KV   | Split | Beam | Align | Long   | Timestamps
               | Graphs| Type         | Cache| Graph | Srch | DTW   | Audio  |
---------------+-------+--------------+------+-------+------+-------+--------+-----------
NeMo TDT       |   2   | RNN-T stream |  NO  |  NO   |  NO  |  NO   | native | token-level
MedASR CTC     |   1   | None (CTC)   |  NO  |  NO   |  NO  |  NO   | native | CTC align
Whisper        |   4   | AR seq2seq   | YES  | YES   | YES  | YES   | 30s win | tokens+DTW
```

## Complexity Scoring

```
Factor                     | NeMo TDT | MedASR | Whisper
---------------------------+----------+--------+--------
ONNX session management    |    2     |   1    |    4
KV cache state machine     |    0     |   0    |    1
Split graph key remapping  |    0     |   0    |    1
Autoregressive decode loop |    0     |   0    |    1
Beam search                |    0     |   0    |    1
Cross-attention alignment  |    0     |   0    |    1
Timestamp token processing |    0     |   0    |    1
Chunking (30s windows)     |    0     |   0    |    1
Logit suppression pipeline |    0     |   0    |    1
Prompt token construction  |    0     |   0    |    1
fp16 tensor handling       |    1     |   0    |    1
---------------------------+----------+--------+--------
TOTAL                      |    3     |   1    |   13
executor.ts lines          |  867     |  871   | 1412
Total model files          |   11     |  10    |   16
```

## The 5 Compounding Factors

### 1. KV Cache State Machine

NeMo TDT and MedASR have NO KV cache. The encoder runs once, the output is processed
directly. No state management.

Whisper has per-layer self-attention + cross-attention KV cache that must be:
- Initialized from decoder_init outputs (`present.*`)
- Remapped to decoder_step inputs (`past_key_values.*`)
- Encoder cross-attention KV preserved across steps (decoder_step doesn't output it)
- Decoder self-attention KV updated each step

This alone accounts for ~200 lines of careful tensor management.

### 2. Autoregressive Decode Loop

NeMo TDT: token-by-token streaming (RNN-transducer — the model is DESIGNED for streaming).
MedASR: no decoder loop at all (CTC is single-pass on encoder output).

Whisper: explicit N-step autoregressive loop with:
- EOS detection (token 50257)
- max_new_tokens cap
- Logit processing BEFORE argmax
- KV cache passing between steps
- Different behavior for first token (init) vs subsequent (step)

### 3. Dual Decoder Path

Whisper has TWO decoder architectures that must coexist in the same executor:

**Merged decoder** (decoder_model_merged.onnx):
- Single graph, `use_cache_branch` boolean flag
- Cross-attention available for alignment
- ~313 lines in executor

**Splitgraph decoder** (4 separate graphs):
- encoder + decoder_init + decoder_step + decoder_align
- Key remapping between graphs
- Encoder KV preservation
- ~149 lines in executor

Every feature (beam search, alignment, timestamps) needs to work in BOTH paths.
The `transcribe()` method dispatches based on which artifacts are loaded.

### 4. Logit Processing Pipeline

NeMo TDT: argmax + blank detection. Done.
MedASR: CTC collapse. Done.
Whisper: suppress_tokens (every step) + begin_suppress_tokens (first step only)
+ WhisperTimestampLogitsProcessor (monotonic timestamps, no_timestamps mode, pair validity)
+ no_speech detection + optional temperature fallback

### 5. Multiple Timestamp Systems

NeMo TDT: timestamps are inherent in the streaming token output.
MedASR: CTC alignment is built into the model architecture.

Whisper has TWO independent timestamp systems:
1. **Timestamp tokens** (50364-51865) — predicted by the decoder like regular tokens
2. **Cross-attention DTW alignment** — via decoder_align.onnx, a separate ONNX graph

Both must produce consistent timing. Both must be tested separately.

---

## Code Distribution in Whisper executor.ts

```
transcribe() main dispatch         294 lines (20.8%)  ← dual-path dispatch + segment building
helpers/utils                      228 lines (16.1%)  ← asset loading, error handling
computeAttentionWordTimestamps     168 lines (11.9%)  ← merged-path alignment
transcribeWithSplitGraph()         141 lines (10.0%)  ← splitgraph decode + segment building
initialize                          84 lines ( 5.9%)  ← session creation (4 sessions)
runDecoderStep (merged)             77 lines ( 5.5%)  ← merged decoder step
runForcedAlignment (merged)         68 lines ( 4.8%)  ← merged alignment
asset/materialize                   66 lines ( 4.7%)  ← HF artifact resolution
beam search                         57 lines ( 4.0%)  ← beam expansion/scoring
computeAttentionWordTimestampsSG    56 lines ( 4.0%)  ← splitgraph alignment
transcribeLongAudio                 52 lines ( 3.7%)  ← 30s chunking
runDecoderStepSplit                 41 lines ( 2.9%)  ← splitgraph decoder step
runDecoderInit                      32 lines ( 2.3%)  ← splitgraph decoder init
runForcedAlignmentSplitGraph        20 lines ( 1.4%)  ← splitgraph alignment
loadGenerationConfig                14 lines ( 1.0%)  ← config loading
loadModelConfig                     14 lines ( 1.0%)  ← config loading
```

The largest blocks are NOT the decode loop — they're the dispatch, segment building,
and alignment code. The actual "run ONNX and get tokens" is the smaller part.

---

## Recommended Approach for the Porting Agent

### Phase 1: Splitgraph-only greedy decode (ALREADY DONE)

This is the simplest path and is already working:
- encoder → decoder_init → decoder_step loop → EOS
- No beam search, no alignment, no chunking
- Validated: fp32/fp16 exact parity on 16 fixtures

### Phase 2: Add logit processing (ALREADY DONE)

- WhisperTimestampLogitsProcessor
- suppress_tokens / begin_suppress_tokens
- no_timestamps mode

### Phase 3: Add alignment (ALREADY DONE)

- decoder_align.onnx session
- processSplitGraphAlignment() DTW
- Word timestamps via cross-attention

### Phase 4: WebGPU smoke (CURRENT TASK)

The WebGPU smoke should ONLY test what's already working:
1. Load fp16 model in browser (4 sessions)
2. Run encoder on mel features
3. Run decoder_init with prompt
4. Run decoder_step loop to EOS
5. Compare tokens with Node/CPU reference

DO NOT add new features during WebGPU smoke. Test what works first.

### Phase 5: Beam search (DEFERRED)

### Phase 6: Language detection (DEFERRED)

### Phase 7: Chunking / long audio (DEFERRED)

### Phase 8: Merged decoder path (LOW PRIORITY)

The merged decoder path exists for backward compat with non-split ONNX exports.
The splitgraph path is the primary target. The merged path can be added later
if needed for compatibility with standard HF ONNX exports.

---

## What NOT to Do

1. **Don't try to implement all features at once.** Whisper has 13 complexity points.
   Implement them one at a time with validation between each.

2. **Don't modify the ONNX export.** The 4-graph split is correct and validated.
   Focus on the runtime, not the exporter.

3. **Don't optimize prematurely.** Session reuse, batched inference, streaming —
   these are all premature until basic decode is correct.

4. **Don't mix merged and splitgraph concerns.** If working on splitgraph,
   don't touch merged decoder code. If working on WebGPU, don't add beam search.

5. **Don't create new test fixtures without curation.** The existing 7 wav fixtures
   cover the needed cases. New fixtures should be reviewed before committing.

6. **Don't touch q8.** q8 is experimental/diagnostic. fp32/fp16 is the target.

---

## Key Reference Files

| Concern | File | Lines |
|---------|------|-------|
| Decode loop (pure) | `executor.ts` L172-228 `splitGraphDecodeLoop` | 57 |
| Splitgraph transcribe | `executor.ts` L1220-1360 `transcribeWithSplitGraph` | 141 |
| KV cache shapes | `executor.ts` L141-170 `computeEmptyPastKeyValueShapes` | 30 |
| Logit processing | `processors.ts` WhisperTimestampLogitsProcessor | 105 |
| V2 validator (reference) | `tests/smoke/whisper-splitgraph-node-wasm-validate-v2.mjs` | 875 |
| WebGPU smoke | `tests/smoke/whisper-webgpu-smoke.html` | 346 |
| WebGPU practical notes | `docs/references/whisper-webgpu-smoke-notes.md` | 224 |
| Reference decode patterns | `docs/references/whisper-reference-decode-patterns.md` | 795 |
