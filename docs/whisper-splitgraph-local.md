# Self-Exported Whisper Splitgraph Models

Load Whisper models exported by `tools/whisper-onnx-export/export_whisper.py` directly from a
local directory in Node.js, or serve from a static URL in browser apps.

## Quick start

```bash
# 1. Export a model (Python)
cd tools/whisper-onnx-export
.venv/bin/python export_whisper.py openai/whisper-tiny /tmp/whisper-tiny-4graph

# 2. Transcribe (Node.js)
cd ../..
WHISPER_MODEL_DIR=/tmp/whisper-tiny-4graph \
  node --experimental-vm-modules examples/whisper-splitgraph-local.mjs
```

## Node.js local usage

```typescript
import {
  createWhisperSeq2SeqModelFamily,
  loadSplitGraphLocalModel,
} from '@asrjs/speech-recognition/models/whisper-seq2seq';

// Read manifest.json + build artifact source from local directory
const { source, config, modelId } = loadSplitGraphLocalModel('/path/to/exported/whisper-tiny');

const factory = createWhisperSeq2SeqModelFamily();
const model = await factory.createModel(
  { modelId, options: { source, config } },
  { backend: { id: 'wasm' }, hooks: {} },
);
const session = await model.createSession();

// Text only
const result = await session.transcribe(audio, { language: 'en' });
console.log(result.utteranceText);

// With word timestamps (requires decoder_align.onnx)
const words = await session.transcribe(audio, {
  language: 'en',
  detail: 'words',
  returnTimestamps: 'word',
});
for (const w of words.words ?? []) {
  console.log(`[${w.startTime.toFixed(2)}s] ${w.text}`);
}

await session.dispose();
await model.dispose();
```

## Required files

Every exported directory must contain:

| File                     | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `manifest.json`          | Model metadata (format, dimensions, alignment heads, special tokens) |
| `encoder_model.onnx`     | Mel spectrogram → encoder hidden states                              |
| `decoder_init.onnx`      | Prompt tokens + hidden states → first logits + full KV cache         |
| `decoder_step.onnx`      | Single token + KV cache → next logits + updated KV cache             |
| `decoder_align.onnx`     | All tokens + hidden states → cross-attention alignment matrix        |
| `tokenizer.json`         | BPE tokenizer vocabulary and merges                                  |
| `generation_config.json` | Alignment heads, suppression rules                                   |
| `config.json`            | Model architecture (optional; manifest provides dimensions)          |

## Manifest format

The `manifest.json` uses format `"whisper-browser-self-export-v1"` and MUST include:

```json
{
  "format": "whisper-browser-self-export-v1",
  "model_id": "openai/whisper-tiny",
  "d_model": 384,
  "decoder_layers": 4,
  "decoder_attention_heads": 6,
  "head_dim": 64,
  "num_mel_bins": 80,
  "max_source_positions": 3000,
  "max_target_positions": 448,
  "vocab_size": 51865,
  "opset": 17,
  "alignment_heads": [[2,2],[3,0],[3,2],[3,3],[3,4],[3,5]],
  "special_tokens": {
    "eos_token_id": 50257,
    "bos_token_id": 50257,
    "pad_token_id": 50257,
    "decoder_start_token_id": 50258,
    "no_timestamps_token_id": 50363,
    "timestamp_begin": 50364,
    "suppress_tokens": [1,2,7,8,...],
    "begin_suppress_tokens": [220,50257]
  },
  "artifacts": {
    "encoder": "encoder_model.onnx",
    "decoder_init": "decoder_init.onnx",
    "decoder_step": "decoder_step.onnx",
    "decoder_align": "decoder_align.onnx"
  }
}
```

### Artifact compatibility checklist

- [ ] `format` = `"whisper-browser-self-export-v1"`
- [ ] `d_model`, `decoder_layers`, `decoder_attention_heads`, `head_dim` present
- [ ] `d_model % decoder_attention_heads == 0`
- [ ] `special_tokens` object present
- [ ] `alignment_heads` present (or baked into decoder_align via averaging)
- [ ] `vocab_size` matches tokenizer.json vocabulary
- [ ] `max_source_positions` typically 3000 (1500 encoder output frames after 2× downsampling)
- [ ] `max_target_positions` typically 448
- [ ] All artifact filenames exist on disk (or at served URLs)

## Browser usage

`file://` URLs are NOT supported in browsers. Serve the exported directory over HTTP(S):

```typescript
// Browser: construct URLs from your static asset server
const source: WhisperArtifactSource = {
  kind: 'splitgraph',
  artifacts: {
    encoderUrl: 'https://example.com/models/tiny/encoder_model.onnx',
    decoderInitUrl: 'https://example.com/models/tiny/decoder_init.onnx',
    decoderStepUrl: 'https://example.com/models/tiny/decoder_step.onnx',
    decoderAlignUrl: 'https://example.com/models/tiny/decoder_align.onnx',
    tokenizerUrl: 'https://example.com/models/tiny/tokenizer.json',
    manifestUrl: 'https://example.com/models/tiny/manifest.json',
  },
};
```

The `loadSplitGraphLocalModel()` helper uses `fs` and `file://` — it is intended for
**Node.js development only**. Browser apps should construct URLs manually or use a
custom URL-based helper.

### Built-in 4-graph WebGPU preset

The browser/WebGPU validation path uses the custom 4-graph repository:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
```

The default WebGPU-friendly splitgraph pairing is:

| Component     | Variant                          |
| ------------- | -------------------------------- |
| Encoder       | `fp16_iofp32/encoder_model.onnx` |
| Decoder init  | `fp16/decoder_init.onnx`         |
| Decoder step  | `fp16/decoder_step.onnx`         |
| Decoder align | `fp16/decoder_align.onnx`        |

If browser logs mention `onnx-community/whisper-large-v3-turbo`, the app is not
exercising this custom 4-graph preset.

When serving very large local ONNX files from the same origin, ORT can consume
the `/models/...` URLs directly. Avoid forcing those local files through
Blob/IndexedDB materialization unless you specifically need cache semantics.

### Whisper mel performance

`WhisperMelProcessor` preserves OpenAI Whisper's `n_fft=400` STFT. It should not
reuse the 512-point NeMo/Parakeet mel processor directly because that changes
the frequency grid. The optimized Whisper implementation uses cached Bluestein
FFT work buffers for the exact 400-point transform.

Benchmark:

```bash
npm run benchmark:whisper-mel
```

Expected ballpark on the 2026-06-14 Windows test host: about `200ms` for 30s of
128-bin Whisper mel features.

## Output modes

| Mode            | Option                                          | Returns                                                                           |
| --------------- | ----------------------------------------------- | --------------------------------------------------------------------------------- |
| Text only       | `{}` (default)                                  | `utteranceText` string                                                            |
| Segments        | `{ detail: "segments" }`                        | `segments[{ startTime, endTime, text, confidence }]`                              |
| Word timestamps | `{ detail: "words", returnTimestamps: "word" }` | `words[{ startTime, endTime, text, confidence }]` (requires `decoder_align.onnx`) |

Word timestamps use cross-attention DTW alignment via `decoder_align.onnx`.
If `decoder_align.onnx` is not available, word timestamps fall back to
timestamp-token interpolation (less accurate).

## Verification env vars

These optional environment variables enable fixture-based tests:

| Var                              | Test                                            | What it verifies                                         |
| -------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| `WHISPER_SPLITGRAPH_FIXTURE_DIR` | `tests/whisper-splitgraph-smoke.test.ts`        | Encoder shape, init→step loop, alignment shape, row sums |
| `WHISPER_REFERENCE_JSON`         | `tests/whisper-reproducibility-harness.test.ts` | Token match vs PyTorch/ONNX Python reference             |

```bash
# Export model + generate reference
cd tools/whisper-onnx-export
.venv/bin/python export_whisper.py openai/whisper-tiny /tmp/tiny
.venv/bin/python generate_hf_reference.py \
  --model-dir /tmp/tiny \
  --audio ../../tools/data/fixtures/audio/jfk-short.wav \
  --output /tmp/ref.json --export-mel

# Run verification
cd ../..
WHISPER_SPLITGRAPH_FIXTURE_DIR=/tmp/tiny \
  npx vitest run tests/whisper-splitgraph-smoke.test.ts

WHISPER_REFERENCE_JSON=/tmp/ref.json \
  npx vitest run tests/whisper-reproducibility-harness.test.ts
```

## Architecture

```
Audio → WhisperMelProcessor → mel spectrogram
  → encoder_model.onnx → hidden_states [1, 1500, d_model]
  → decoder_init.onnx(prompt) → logits + KV cache
  → decoder_step.onnx(token, KV) × N → tokens
  → decoder_align.onnx(all_tokens) → alignment [1, T, 1500]
  → processSplitGraphAlignment() → DTW → word timestamps
```

Output is identical to merged-decoder path (`WhisperNativeTranscript`) with same
word timestamp schema.
