# Whisper WebGPU Smoke — Session & Inference Notes

Practical notes for fp16 WebGPU smoke testing in the browser.

**Target**: whisper-base fp16 4-graph ONNX via onnxruntime-web WebGPU backend.

---

## Model Files Needed

From `/tmp/whisper-base-4graph/fp16/` (or HF `ysdede/whisper-base-onnx-4graph` fp16 variant):

```
encoder_model.onnx    40 MB
decoder_init.onnx    150 MB
decoder_step.onnx     94 MB
decoder_align.onnx    95 MB  (optional, for alignment only)
manifest.json
tokenizer.json
generation_config.json
config.json
```

Total: ~384 MB inline (no external data for whisper-base).

---

## ONNX Runtime WebGPU Session Options

```js
const sessionOptions = {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all',   // or 'extended' for faster creation
};
```

### Fallback Pattern

```js
let ep = 'webgpu';
if (typeof navigator !== 'undefined' && !navigator.gpu) {
  ep = 'wasm';  // fallback to WASM CPU
}
```

### Session Creation Cost

- WebGPU sessions trigger shader compilation on creation
- whisper-base encoder (~40MB): ~1-3s on decent GPU
- whisper-base decoder_init (~150MB): ~3-8s
- whisper-base decoder_step (~94MB): ~2-5s
- Total session creation: ~10-20s on first run (shader cache helps on repeat)

### Session Reuse

Sessions MUST be reused across multiple transcriptions. Do NOT create per-request:
- Create once on page load
- Reuse for all subsequent runs
- Only destroy on page unload

---

## fp16 Tensor Handling

### Input: fp16 mel features

```js
// Model expects float16 input for fp16 variant
const melFp16 = float32ToFloat16Bits(melFloat32);
const melTensor = new ort.Tensor('float16', melFp16, [1, 80, 3000]);
```

### Encoder Output: fp16 hidden states

```js
const encOut = await encSess.run({ input_features: melTensor });
const encData = encOut[Object.keys(encOut)[0]];
// encData.type === 'float16', encData.data is Uint16Array
// Pass directly to decoder — no conversion needed
```

### Decoder Logits: fp16 → float32 for argmax

```js
// CRITICAL: fp16 logits must be converted to float32 before argmax
const logits = ensureFloat32(stepOut[logitsKey].data, stepOut[logitsKey].type);
const nextToken = argmax(logits);
```

**Why**: Raw fp16 half bits are NOT numeric logits. `argmax` on `Uint16Array` compares
integer values, not float values. This produces garbage tokens. Always convert to float32
before any numerical comparison.

### Alignment Output: same fp16 → float32 conversion needed

```js
const alignLogits = ensureFloat32(alignOut[...].data, alignOut[...].type);
// Row sums and DTW require float32 values
```

---

## Decode Loop Reference

### Prompt Construction

```js
const PROMPT_IDS = [50258, 50259, 50359, 50363];
// <|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>
```

### Full Loop (greedy, no timestamp/logit processing for smoke)

```js
// 1. Encoder
const melTensor = new ort.Tensor('float16', melFp16, [1, 80, 3000]);
const encOut = await encSess.run({ input_features: melTensor });
const encData = encOut[Object.keys(encOut)[0]];

// 2. Decoder init
const promptTensor = new ort.Tensor('int64',
  new BigInt64Array(PROMPT_IDS.map(BigInt)), [1, PROMPT_IDS.length]);
const initOut = await initSess.run({
  input_ids: promptTensor,
  encoder_hidden_states: encData
});

// 3. First token from init logits
const initKeys = Object.keys(initOut);
const logitsKey = initKeys.find(k => k.includes('logits')) ?? initKeys[0];
const initLogits = ensureFloat32(initOut[logitsKey].data, initOut[logitsKey].type);
const vocabSize = initOut[logitsKey].dims.at(-1);
const firstLogits = initLogits.subarray(initLogits.length - vocabSize);
let nextToken = argmax(firstLogits);
const tokens = [nextToken];

// 4. Build KV cache (present.* → past_key_values.*)
const pastKv = {};
for (const key of initKeys) {
  if (key.startsWith('present'))
    pastKv[key.replace(/^present\./, 'past_key_values.')] = initOut[key];
}

// 5. Step loop
let eosReached = nextToken === 50257;
for (let step = 1; step < MAX_NEW_TOKENS && !eosReached; step++) {
  const feeds = {
    input_ids: new ort.Tensor('int64',
      new BigInt64Array([BigInt(nextToken)]), [1, 1])
  };
  for (const [k, v] of Object.entries(pastKv)) feeds[k] = v;

  const stepOut = await stepSess.run(feeds);
  const stepKeys = Object.keys(stepOut);
  const stepLogitsKey = stepKeys.find(k => k.includes('logits')) ?? stepKeys[0];
  const stepLogits = ensureFloat32(stepOut[stepLogitsKey].data, stepOut[stepLogitsKey].type);
  nextToken = argmax(stepLogits);
  tokens.push(nextToken);
  eosReached = nextToken === 50257;

  // Update KV: step outputs self-attn only, preserve encoder KV
  for (const key of stepKeys) {
    if (key.startsWith('present'))
      pastKv[key.replace(/^present\./, 'past_key_values.')] = stepOut[key];
  }
}
```

---

## KV Cache Key Remapping

This is the #1 source of bugs in the 4-graph architecture:

```
decoder_init outputs:  present.{i}.decoder.key, present.{i}.encoder.key
decoder_step expects:  past_key_values.{i}.decoder.key, past_key_values.{i}.encoder.key
```

**Must remap**: `key.replace(/^present\./, 'past_key_values.')`

### Encoder KV Preservation

`decoder_step` outputs ONLY self-attention (decoder-side) KV. Encoder cross-attention KV
from `decoder_init` must be preserved:

```js
// After step, only update decoder-side keys.
// Encoder keys remain from init and are never overwritten.
for (const key of stepKeys) {
  if (key.startsWith('present')) {
    pastKv[key.replace(/^present\./, 'past_key_values.')] = stepOut[key];
  }
}
// pastKv still contains encoder.* keys from init — untouched
```

### KV Cache Shapes (whisper-base fp16)

```
Self-attn KV per layer:  [1, 8, text_len, 64]    — 8 heads, head_dim=64
Cross-attn KV per layer: [1, 8, 1500, 64]         — frozen from encoder
Total layers: 6 (decoder)
Total KV entries: 6 layers × 2 (K,V) × 2 (self + cross) = 24 tensors
```

---

## Known Issues / Pitfalls

### 1. ORT WebGPU version

ORT WebGPU support varies by version. The smoke HTML uses `ort.all.min.js` from CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.min.js"></script>
```
Verify this version supports all ops in the whisper-base graphs. If ops are missing, try newer.

### 2. Float16 output from encoder

WebGPU backend may output fp16 tensors even when the ONNX graph declares float32 output.
Always use `ensureFloat32()` before numerical operations.

### 3. Large model loading

whisper-base fp16 is ~384MB. Browser must download all ONNX files via HTTP.
Ensure:
- HTTP server sends correct MIME types (`application/octet-stream` for `.onnx`)
- CORS headers not needed for same-origin serving
- Sufficient browser memory (Chrome needs ~2x model size in WASM memory)

### 4. No logit processing in smoke

The smoke test runs greedy decode WITHOUT:
- SuppressTokensLogitsProcessor
- WhisperTimeStampLogitsProcessor
- Temperature / sampling

This is intentional for minimal smoke. Production needs full logit processing.

### 5. Audio decoding in browser

`AudioContext.decodeAudioData()` may resample. Verify output is 16kHz mono.
The HTML smoke handles this with linear resampling fallback.

---

## Reference Output (whisper-base fp16, jfk2.en.wav)

Expected greedy decode output (from V2 Node/CPU validator):

```
Tokens: [400, 370, 452, 7177, 6280, 11, 1029, 406, 437, 428, 1941, 393, 360, 337, 291, 11, 1029, 437, 291, 393, 360, 337, 428, 1941, 13, 50257]
Text: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
Token count: 26 (including EOS)
EOS reached: true
```

WebGPU output should match exactly for fp16. Any token difference is a WebGPU runtime bug,
not a quantization issue (fp16 is exact-match with fp32 on Node/CPU).

---

## Next Steps After fp16 Smoke Passes

1. Add logit processors (suppress_tokens, timestamp rules)
2. Test longer audio (near 30s boundary)
3. Test Turkish fixture with `PROMPT_IDS = [50258, 50268, 50359, 50363]`
4. Test with alignment (decoder_align.onnx)
5. Consider large-v3-turbo fp16 (needs external data support in browser)
