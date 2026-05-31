# Multi-Backend Decoding Policy Verification

## When to use

One backend (e.g., Node ORT fp32) produces a correct transcript, another (e.g., WebGPU fp16/fp16io) produces early EOS or empty transcript with the same model and audio. **Before investigating model precision**, verify the generation policy is identical across backends.

## Policy checklist

### 1. Prompt construction

Compare the prompt tokens between the working and failing backend:

| Token | ID (large-v3-turbo) | Purpose |
|-------|---------------------|---------|
| `<\|startoftranscript\|>` | 50258 | SOT |
| `<\|en\|>` | 50259 | Language |
| `<\|transcribe\|>` | 50360 | Task |
| `<\|notimestamps\|>` | 50364 | no_timestamps flag |

**Runner (executor.ts:1307-1315)**: `[50258, 50259, 50360, 50364]` — all 4 tokens.

**Common test-page mistake**: only `[50258, 50259]` (SOT + lang). When the task and notimestamps tokens are NOT in the prompt, the decoder generates them as **regular tokens**:
- Step 0: 50360 (transcribe) — not a text token, consumed as special
- Step 1: 50364 (notimestamps) — consumed as special
- Step 2: EOS fires (only real decode step)
- Result: "empty transcript" even though logits were valid at every step

The token sequence `50360 → 50364 → EOS` is diagnostic of a **missing prompt tokens** problem, NOT a model precision problem.

### 2. suppress_tokens

From `generation_config.json` — a list of token IDs that are ALWAYS set to `-Infinity` at every step. These are special tokens that should never appear in output:

```
[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60,
 61, 62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893,
 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253,
 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061,
 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157,
 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675,
 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
 36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361,
 50362, 50363]
```

**Check**: does the failing backend apply `suppress_tokens` from the same list?

### 3. begin_suppress_tokens

From `generation_config.json` — **`[220, 50257]`** (EOS = 50257). These tokens are suppressed ONLY on the first free text-generation step (step 0 of the decode loop, after decoder_init).

**How it fires:**

```typescript
// In WhisperTimestampLogitProcessor.process():
// generatedTokens = full prompt (e.g. [50258, 50259, 50360, 50364])
// beginIndex = promptTokens.length (e.g. 4)
if (generatedTokens.length === beginIndex) {
  // ONLY at step 0 of the decode loop
  for (const tokenId of this.beginSuppressTokens) {
    logits[tokenId] = -Infinity; // EOS = -Infinity at step 0
  }
}
```

**Critical**: `begin_suppress_tokens` only fires ONCE. At all subsequent steps, EOS is available.

**Check output**:
- Step 0 logits: log EOS logit BEFORE and AFTER suppression
  - Before: some value (7.73 or similar)
  - After: -Infinity (EOS blocked)
- Step 1+ logits: EOS should be available (not -Infinity)

If step 0 already has EOS selected (without suppression), it's a begin_suppress_tokens bug. If step 1+ has EOS with 0.03 higher than text, it's an encoder precision issue.

### 4. forced_decoder_ids

`generation_config.json: forced_decoder_ids: [[1, null], [2, 50360]]`

- Step 1 (first prompt position): auto-detect language (null = not forced)
- Step 2 (second prompt position): force transcribe token (50360)

The runner does NOT use forced_decoder_ids; it manually constructs the prompt with all tokens. If a test page omits these, the decoder will generate them as normal tokens, wasting decode steps and potentially triggering early EOS.

### 5. EOS handling

- `eos_token_id`: 50257
- `bos_token_id`: 50257 (same ID, dual role)
- `pad_token_id`: 50257

The runner checks `nextTokenId === eosTokenId` after each step and breaks on match. The processor applies `begin_suppress_tokens` which includes EOS at step 0.

### 6. Greedy vs sampling

Both backends should use argmax (greedy, temperature=0). No randomness. If one backend uses sampling (temperature > 0), results differ by definition and are not comparable.

### 7. KV cache management (splitgraph only)

The split decoder step model outputs ONLY decoder self-attention KV caches. Encoder cross-attention KV must be preserved:

- decoder_init outputs: `present.0.decoder.key`, `present.0.decoder.value`, `present.0.encoder.key`, `present.0.encoder.value`
- decoder_step outputs: `present.0.decoder.key`, `present.0.decoder.value` (NO encoder KV)
- After each step, merge decoder KV from output with encoder KV from previous cache:

```javascript
const oldPkv = {...pkv};
// Update decoder KV from step output (rename present. → past_key_values.)
for (const k of stepKeys) {
  if (k.startsWith('present'))
    pkv[k.replace(/^present\./, 'past_key_values.')] = stepOutput[k]; }
// Preserve encoder KV (step model doesn't output it):
for (const [k, v] of Object.entries(oldPkv)) {
  if (k.includes('encoder') && !pkv[k]) pkv[k] = v; }
```

**Error if missing**: ORT throws `invalid input 'past_key_values.0.encoder.key'` at decoder_step step 2+.

## Verification procedure

When a backend variant produces early EOS:

```
1. DUMP the full token sequence (all generated token IDs)
   - If sequence includes task/notimestamps tokens as generated tokens → prompt is too short
   - Fix: add missing tokens to prompt

2. DUMP logits at step 0 before/after suppression
   - Verify EOS is actually -Infinity after begin_suppress_tokens
   - If EOS is still finite → begin_suppress_tokens not wired

3. DUMP logits at step 2 (where EOS often fires)
   - Compare EOS logit vs top text logit
   - If EOS > text by <0.1 AND begin_suppress_tokens is correct → encoder distribution shift

4. If policy is correct → compare encoder outputs
   - fp32 encoder (working) vs fp16io encoder (failing)
   - Cosine similarity between output tensors
   - Per-token min/max/mean statistics
   - Per-channel distribution comparison over hidden dim 1280
```

## Encoder output comparison (after policy verified)

```python
import onnxruntime as ort
import numpy as np

# Same mel input
mel = np.load('jfk_mel.npy').astype(np.float32)

# fp32 encoder
sess_fp32 = ort.InferenceSession('fp32/encoder_model.onnx')
out_fp32 = sess_fp32.run(None, {'input_features': mel})[0]

# fp16io encoder (fp16 internal, fp32 I/O)
sess_fp16 = ort.InferenceSession('fp16io/encoder_model.onnx')
out_fp16 = sess_fp16.run(None, {'input_features': mel})[0]

# Statistics
print('fp32: min={} max={} mean={}'.format(
    out_fp32.min(), out_fp32.max(), out_fp32.mean()))
print('fp16: min={} max={} mean={}'.format(
    out_fp16.min(), out_fp16.max(), out_fp16.mean()))

# Cosine similarity
dot = np.sum(out_fp32 * out_fp16)
norm_a = np.linalg.norm(out_fp32)
norm_b = np.linalg.norm(out_fp16)
cos_sim = dot / (norm_a * norm_b)
print('cosine similarity:', cos_sim)

# Per-token (over sequence dimension, axis=1)
seq_sim = np.sum(out_fp32 * out_fp16, axis=2) / (
    np.linalg.norm(out_fp32, axis=2) * np.linalg.norm(out_fp16, axis=2))
print('mean per-token cos sim:', seq_sim.mean())
print('min per-token cos sim:', seq_sim.min())

# Per-channel (over hidden_dim, axis=2)
channel_means_fp32 = out_fp32.mean(axis=(0, 1))
channel_means_fp16 = out_fp16.mean(axis=(0, 1))
channel_shift = channel_means_fp16 - channel_means_fp32
print('channel shift: mean={} std={} max_abs={}'.format(
    channel_shift.mean(), channel_shift.std(), np.abs(channel_shift).max()))
```

If cosine similarity < 0.995 or per-token min < 0.95, the fp16 encoder compute is measurably shifting the distribution. The decoder then sees different context → different confidence → early EOS.
