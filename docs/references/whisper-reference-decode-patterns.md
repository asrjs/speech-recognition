# Whisper Reference Decode Patterns

Comparative study of OpenAI Whisper, HF Transformers, faster-whisper, and whisper.cpp
decode loops, prompt construction, KV cache, logit processing, and session lifecycle.

**Purpose**: Reference guide for asrjs/speech-recognition ONNX WebGPU/WASM 4-graph runtime.

---

## 1. Executive Summary

All four implementations share the same fundamental Whisper architecture:
- **Mel input**: 80 bands, 3000 frames (30s), hop=160, n_fft=400
- **Encoder output**: (batch, 1500, dim) — conv layers downsample 2x
- **Prompt tokens**: [SOT=50258, lang_token, task_token, no_timestamps=50363]
- **EOS**: token 50257
- **Timestamp tokens**: 50364–51865
- **KV cache**: cross-attention frozen after encoder, self-attention grows incrementally
- **Max text context**: 448 tokens

Key differences relevant to asrjs:

| Aspect | OpenAI | HF Transformers | faster-whisper | whisper.cpp |
|--------|--------|----------------|----------------|-------------|
| Runtime | PyTorch | PyTorch | CTranslate2 (C++) | GGML (C) |
| Beam search | Custom | GenerationMixin | CTranslate2 native | Manual C |
| KV cache | Manual dict | past_key_values tuple | Opaque C++ | Explicit structs |
| VAD | None | None | Silero VAD | None |
| Long audio | 30s windows | chunk_generate | VAD + 30s windows | 30s sliding window |
| Language detect | Single decoder pass | Same | Same | Same |
| Timestamp rules | Inline function | LogitsProcessor class | CTranslate2 internal | Inline C |

---

## 2. Prompt Construction

### Token ID Constants (shared across all implementations)

```
SOT              = 50258  <|startoftranscript|>
EOS              = 50257  <|endoftext|>
TRANSCRIBE       = 50359  <|transcribe|>      (NOTE: some refs use 50360)
TRANSLATE        = 50358  <|translate|>
NO_TIMESTAMPS    = 50363  <|notimestamps|>
NO_SPEECH        = 50362  <|nospeech|>
TIMESTAMP_BEGIN  = 50364  <|0.00|>
TIMESTAMP_END    = 51865  <|30.00|>
LANG_EN          = 50259  <|en|>
LANG_TR          = 50268  <|tr|>
```

### OpenAI Whisper (decoding.py)

```python
# DecodingTask builds the initial token sequence:
# [SOT, language_token, task_token, no_timestamps_token]
# With timestamps enabled, no_timestamps is omitted

# For English transcription:
tokens = [50258, 50259, 50359, 50363]

# For Turkish transcription:
tokens = [50258, 50268, 50359, 50363]

# For translation (any language → English):
tokens = [50258, lang_token, 50358, 50363]
```

### HF Transformers (generation_whisper.py)

```python
# Uses forced_decoder_ids:
# Position 0: forced_bos_token_id = 50258 (SOT)
# Position 1: language token (e.g., 50259 for en)
# Position 2: task token (50359 transcribe, 50358 translate)
# Position 3: no_timestamps token (50363)

# The generate() framework handles these via decoder_input_ids
# built from forced_bos_token_id and forced_decoder_ids
```

### asrjs Notes

- Current asrjs uses: `[50258, lang_token, 50360, 50364]`
- **Verified**: tokenizer.json returns TRANSCRIBE=50359, NO_TIMESTAMPS=50363, TIMESTAMP_BEGIN=50364
- `src/` runtime fallbacks match: `?? 50359`, `?? 50363`, `?? 50364` ✓
- **V2 validator fallback bug** (cosmetic, not runtime): uses `?? 50360` and `?? 50364` for transcribe/notimestamps
  — the fallback is never reached because `getTokenId` always resolves. Should be fixed to `50359`/`50363`.
- The prompt must be identical across fp32/fp16/q8 for fair comparison (already enforced)

---

## 3. Language Detection (Auto)

All four implementations use the same algorithm:

```python
# 1. Encode mel features → encoder_output
# 2. Pass single SOT token (50258) to decoder
# 3. Read logits at position 0
# 4. Language token IDs are 50259..50259+98 (99 languages)
# 5. argmax over language token range → detected language
```

### asrjs Implications

- `language: "auto"` currently falls back to English (`<|en|>`)
- True auto-language requires: encode → single SOT decode → read language logits
- This is a separate feature, not a quick fix
- The 4-graph splitgraph can support this: run encoder + decoder_init with [SOT] only, read language logits, then re-run with full prompt

---

## 4. Decoder Loop

### OpenAI Whisper — Greedy

```python
# decoding.py — GreedyDecoder.run()
def run(tokens, encoder_output):
    for _ in range(max_length):  # max_length = 448
        logits = model.decoder(tokens, encoder_output, kv_cache=kv_cache)
        logits = logits[:, -1]  # last position only

        # Apply suppression
        logits = suppress_tokens(logits, suppress_tokens_mask)
        logits = apply_timestamp_rules(logits, tokens, ...)

        next_token = logits.argmax(dim=-1)
        tokens = cat([tokens, next_token], dim=-1)

        if all_sequences_hit_EOS:
            break
    return tokens
```

### OpenAI Whisper — Beam Search

```python
# decoding.py — BeamSearchDecoder.run()
beam_width = beam_size  # default 5
beams = [(0.0, initial_tokens)]

for step in range(max_length):
    candidates = []
    for score, tokens in beams:
        logits = model.decoder(tokens, encoder_output, kv_cache=...)
        logits = logits[:, -1]
        logits = suppress_tokens(logits)
        log_probs = F.log_softmax(logits, dim=-1)
        topk_probs, topk_ids = log_probs.topk(beam_width * 2)
        for prob, tok in zip(topk_probs, topk_ids):
            candidates.append((score + prob, cat([tokens, tok])))
    beams = sorted(candidates)[:beam_width]
    if all_beams_ended_with_EOS: break

return beams[0][1]  # best beam
```

### HF Transformers

```python
# Uses GenerationMixin._sample() or _beam_search()
# Whisper-specific logits processors registered:
# - SuppressTokensLogitsProcessor(suppress_tokens)
# - SuppressTokensAtBeginLogitsProcessor(begin_suppress_tokens, begin_index=0)
# - WhisperTimeStampLogitsProcessor
# - ClassifierFreeGuidanceLogitsProcessor (optional)

# Standard HF generate loop handles:
# - EOS detection per sequence
# - KV cache passthrough (past_key_values)
# - Beam pruning/expansion
# - Length penalties
```

### whisper.cpp — Core Loop

```c
// whisper_decode_internal() — runs n_text_layer decoder layers
// whisper_process_token() — manages beam expansion, token selection, stopping

// For each step:
// 1. Self-attention over previous tokens (using kv_self cache)
// 2. Cross-attention over encoder output (using kv_cross cache)
// 3. FFN
// 4. Apply repetition penalty, temperature
// 5. Select top-k candidates, expand beams
// 6. Prune to beam_size best
// 7. Continue until EOS or max_len
```

### Temperature Fallback Strategy (OpenAI + faster-whisper + whisper.cpp)

```
If decode quality fails (log_prob_threshold, compression_ratio_threshold):
  Retry with increasing temperature: 0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0
```

This is important for production robustness — asrjs should implement this eventually.

---

## 5. KV Cache Management

### Shape Convention

```
Self-attention KV:  (batch, n_heads, current_text_len, head_dim)
Cross-attention KV: (batch, n_heads, 1500, head_dim)  — frozen after encoder
```

### OpenAI Whisper

- KV stored as simple dict/list, updated in-place
- Cross-attention KV computed on first forward pass, frozen for all subsequent steps
- Self-attention KV grows by 1 in seq dimension per token

### HF Transformers

- `past_key_values`: tuple of tuples, each inner tuple has 4 tensors:
  - `(self_attn_k, self_attn_v, cross_attn_k, cross_attn_v)`
- Shape per tensor: `(batch, n_heads, seq_len, head_dim)`
- For whisper-large-v3:
  - 32 layers, 20 heads, head_dim=64
  - Cross KV total: 32 × 2 × B × 20 × 1500 × 64 ≈ 120MB per batch item

### whisper.cpp

```c
struct whisper_state {
    // Self-attention: [n_layer][2][max_tokens+1][d_model]
    // [2] = K and V
    ggml_tensor * kv_self;

    // Cross-attention: [n_layer][2][n_audio_ctx][d_model]
    // Computed once after whisper_encode(), shared across beams
    ggml_tensor * kv_cross;
};

// Between segments: kv_self cleared, kv_cross kept if same audio
```

### asrjs 4-Graph KV Convention

- **decoder_init** outputs: `present.{i}.decoder.key`, `present.{i}.encoder.key`, etc.
- **decoder_step** expects: `past_key_values.{i}.decoder.key`, `past_key_values.{i}.encoder.key`, etc.
- Key remapping: `key.replace(/^present\./, 'past_key_values.')`
- **Encoder KV preservation**: decoder_step outputs ONLY self-attention KV.
  Encoder cross-attention KV from init must be preserved and merged back.
- This is already implemented and verified in the V2 validator.

---

## 6. Logit Processing / Suppression

### Token Suppression Hierarchy

1. **suppress_tokens** — always suppressed (set to -inf)
   - Applied every step
   - Includes non-speech tokens, partial UTF-8 tokens

2. **begin_suppress_tokens** — suppressed only at step 0
   - Applied on first logit computation
   - Includes blank/noise tokens

3. **Timestamp suppression** (WhisperTimeStampLogitsProcessor rules):
   - If `no_timestamps=true`: suppress ALL timestamp tokens (50364–51865)
   - If timestamps enabled:
     - First token must be a timestamp
     - Timestamps must be monotonically increasing
     - Must come in pairs (opening, closing)
     - Max timestamp ≤ 30.0s
     - Duplicate timestamps suppressed

4. **No-speech detection**:
   - Token 50362 (`<|nospeech|>`) probability checked
   - If `no_speech_prob > no_speech_threshold`: skip segment
   - OpenAI default threshold: 0.6

### asrjs Implementation

- `WhisperTimestampLogitProcessor` in `src/models/whisper-seq2seq/processors.ts`
- Implements: suppress_tokens, begin_suppress_tokens, no_timestamps, monotonic enforcement
- Tests: `tests/whisper-timestamp-processor.test.ts` (9 focused tests)
- **Verified correct** via fp32/fp16 parity validation

---

## 7. EOS and Stopping Criteria

### Stopping Conditions (all implementations)

1. **EOS token** (50257) generated → sequence complete
2. **Max tokens** reached (448 for text context)
3. **No-speech threshold** exceeded (optional)
4. **Compression ratio** exceeded (optional quality check)
5. **Log probability threshold** (optional quality check)

### asrjs Current Behavior

- Greedy decoding stops on EOS or max_new_tokens
- EOS normalization: `[tokens..., EOS]` and `[tokens...]` treated as equivalent
- No compression ratio / log probability quality checks yet (deferred)

---

## 8. Mel Frontend

### Parameters (shared across all implementations)

```
sample_rate  = 16000
n_fft        = 400     (25ms window)
hop_length   = 160     (10ms stride)
n_mels       = 80      (or 128 for some configs)
f_min        = 0
f_max        = 8000
```

### Input/Output Shapes

```
Input audio:   (samples,) at 16kHz
Mel output:    (n_mels, n_frames) where n_frames = audio_length / 160
Padded input:  (1, n_mels, 3000)  — always 3000 frames for 30s
Encoder input: (1, n_mels, 3000)
Encoder output: (1, 1500, d_model)  — conv2 downsamples by 2x
```

### 2x Downsampling in Encoder

```python
# OpenAI Whisper encoder:
x = conv1(mel)   # Conv1d(80, dim, kernel=3, stride=1, pad=1) → (B, dim, 3000)
x = conv2(x)     # Conv1d(dim, dim, kernel=3, stride=2, pad=1) → (B, dim, 1500)
x = x.permute(0, 2, 1)  # → (B, 1500, dim)
```

### asrjs Pitfall (already fixed)

- `config.maxSourcePositions=1500` is encoder OUTPUT positions
- Encoder INPUT expects 3000 mel frames
- When padding: `inputFrames = maxSourcePositions <= 1500 ? maxSourcePositions * 2 : maxSourcePositions`
- This was a real bug — runtime padded to 1500 while validator padded to 3000

---

## 9. Session Lifecycle (ONNX)

### asrjs 4-Graph Architecture

```
encoder_model.onnx   — mel → encoder_hidden_states (run once per audio)
decoder_init.onnx    — prompt + encoder_hidden_states → logits + KV cache
decoder_step.onnx    — token + past KV → logits + updated KV (autoregressive loop)
decoder_align.onnx   — all_tokens + encoder_hidden_states → attention alignment
```

### Session Reuse Pattern (V2 validator)

```js
// Create once per variant:
const sessions = await createSessions(ort, variantDir, manifestRaw, runtimeBackend);
// { encoder, decoderInit, decoderStep, decoderAlign }

// Reuse across all fixtures:
for (const fixture of fixtures) {
  await runFixture(sessions, state, fixture, maxNewTokens, enableAlign);
}

// Release at variant boundary:
releaseSessions(sessions);
```

### V1 vs V2 Performance

| Metric | V1 (per fixture) | V2 (per variant) |
|--------|------------------|-------------------|
| Sessions created | N × V × 3 | V × 3 |
| ONNX disk I/O | ~6 GB (7 fixtures × 2 variants) | ~0.85 GB |
| Wall time (whisper-base) | 2m50s | 1m27s |

### WebGPU Session Considerations

- ONNX Runtime WebGPU sessions have similar creation overhead
- Session creation is async and should be done once
- `graphOptimizationLevel: 'all'` runs optimizer on creation — expensive
- Consider `'basic'` or `'extended'` for development, `'all'` for production
- WebGPU session creation may involve shader compilation — cache where possible

---

## 10. Alignment / Timestamps

### Cross-Attention Alignment (OpenAI / HF)

- Extract cross-attention weights from decoder layers
- Use specific attention heads (alignment_heads from generation_config)
- DTW (Dynamic Time Warping) on attention matrix → word-level timestamps
- Attention matrix shape: `(n_heads, text_tokens, encoder_frames)`

### faster-whisper DTW

- Uses cross-attention from CTranslate2 decoder
- Computes alignment for each word using attention head averaging
- DTW finds optimal path through attention matrix
- Word boundaries extracted from DTW path

### whisper.cpp Timestamps

- Relies on model's timestamp tokens (50364–51865) for segment-level timing
- Token-level timestamps from `<|t.tt|>` tokens in output
- No DTW / word-level alignment built-in

### asrjs Splitgraph Alignment

- `decoder_align.onnx` — dedicated alignment graph
- Input: all tokens + encoder_hidden_states
- Output: attention alignment matrix `[1, T_all, S]`
- `processSplitGraphAlignment()` — slices prompt rows, builds DTW, extracts timestamps
- Already implemented and verified

---

## 11. Chunking / Long Audio

### OpenAI Whisper

- Fixed 30-second windows
- No built-in VAD
- Each window processed independently
- `condition_on_previous_text=True` prepends prior transcript as context

### HF Transformers — `chunk_generate`

```python
# For audio > 30s:
# 1. Split into overlapping 30s windows
# 2. Process each chunk independently
# 3. Merge outputs, handling overlapping timestamps
# 4. WhisperProcessor handles the windowing
```

### faster-whisper — VAD-Based Chunking

```python
# Silero VAD integration:
# 1. Run Silero VAD on full audio → speech segments
# 2. Each speech segment padded by vad_speech_pad_ms (default 400ms)
# 3. Each segment independently transcribed
# 4. Previous text fed as prompt for context coherence

# Without VAD:
# Fixed 30s windows with overlap handling
# Model's timestamp tokens determine speech end within window
```

### whisper.cpp — Sliding Window

```c
// whisper_full():
// 1. Compute mel for entire audio
// 2. Sliding window of 30s (WHISPER_SEGMENT_SIZE)
// 3. Each window: encode → decode → extract timestamps
// 4. Next window starts at end of last detected speech
// 5. split_on_word option for cleaner boundaries
```

### asrjs Long Audio Strategy

- Currently: single 30s window per `transcribe()` call
- Pipeline layer handles windowing via catalog-driven policy
- Whisper uses 30s max, Parakeet TDT uses 90s/180s
- Future: VAD-based chunking similar to faster-whisper

---

## 12. Actionable Recommendations for asrjs ONNX WebGPU/WASM

### Priority 1: fp16 WebGPU Smoke

Use V2 validator's fp32/fp16 Node output as ground truth reference:
- Load whisper-base fp16 model in browser via ORT WebGPU
- Run encoder → decoder_init → decoder_step loop on small fixture set
- Compare token output with V2 Node/CPU reference
- Check: model loading, encoder run, decode loop, EOS behavior, decoded text

### Priority 2: Language Detection

Implement using the 4-graph architecture:
1. Run encoder (reuse session)
2. Run decoder_init with `[SOT]` only → read language logits
3. argmax over 50259..50259+98 → detected language
4. Re-run decoder_init with full prompt including detected language
5. Continue with decoder_step loop

This is clean because decoder_init can be run twice with the same session.

### Priority 3: Beam Search

Follow HF Transformers pattern:
- `BeamSearchDecoder` maintaining beam_width hypotheses
- Per-beam KV cache (self-attention only, cross-attention shared)
- Log probability accumulation + length normalization
- Temperature fallback strategy (0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0)

### Priority 4: Session Lifecycle Optimization

For WebGPU:
- Session creation involves shader compilation — cache aggressively
- Consider `graphOptimizationLevel: 'extended'` for dev, `'all'` for production
- Encoder session can be shared across multiple transcribe() calls for streaming
- decoder_align session can be lazy-loaded (only needed when timestamps requested)

### Priority 5: Quality Checks

Add production robustness features from reference implementations:
- Compression ratio threshold (detect repetitive/degenerate output)
- Log probability threshold (detect low-confidence output)
- No-speech probability check (skip silence segments)
- Temperature fallback on quality failure

---

## Appendix: Model Dimension Reference

| Model | Params | d_model | enc_layers | dec_layers | heads | head_dim | n_mels |
|-------|--------|---------|------------|------------|-------|----------|--------|
| tiny  | 39M    | 384     | 4          | 4          | 6     | 64       | 80     |
| base  | 74M    | 512     | 6          | 6          | 8     | 64       | 80     |
| small | 244M   | 768     | 12         | 12         | 12    | 64       | 80     |
| medium| 769M   | 1024    | 24         | 24         | 16    | 64       | 80     |
| large | 1550M  | 1280    | 32         | 32         | 20    | 64       | 80     |
| large-v3-turbo | 809M | 1280 | 32 | 4 | 20 | 64 | 128 |

All models: head_dim is always 64 (d_model / heads), audio_ctx=1500, text_ctx=448, vocab=51865.
