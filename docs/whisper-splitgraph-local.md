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
} from '@asrjs/speech-recognition/models/whisper-seq2seq';
import { loadSplitGraphLocalModel } from '@asrjs/speech-recognition/models/whisper-seq2seq/local-file';

// Read manifest.json + build artifact source from local directory
const { source, config, modelId } = await loadSplitGraphLocalModel('/path/to/exported/whisper-tiny');

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
  "max_source_positions": 1500,
  "max_target_positions": 448,
  "vocab_size": 51865,
  "opset": 17,
  "alignment_heads": [[2,2],[3,0],[3,2],[3,3],[3,4],[3,5]],
  "alignment_export": {
    "causal_self_attention": true,
    "encoder_hidden_state_dtype": "float16",
    "attention_implementation": "eager",
    "attention_values": "logits",
    "attention_layout": "selected_heads"
  },
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
- [ ] `alignment_heads` present and reflected in the alignment export contract
- [ ] `alignment_export.causal_self_attention` is `true`
- [ ] New exports declare `attention_values: "logits"` and
      `attention_layout: "selected_heads"`; legacy averaged probability graphs
      must declare `post_softmax` and `mean`
- [ ] `vocab_size` matches tokenizer.json vocabulary
- [ ] `max_source_positions` typically 1500; the encoder graph still accepts
      3000 mel frames and emits 1500 encoded positions after 2x downsampling
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

### Decoder performance profile

The 4-graph layout is for both correctness and speed:

- `decoder_init.onnx` runs the prompt prefill and creates the first KV cache.
- `decoder_step.onnx` consumes one new token plus KV cache and emits updated KV.
- KV cache avoids recomputing previous decoder tokens on every step.

It does not make Whisper decoding parallel. Whisper remains autoregressive, so a
50-token transcript still requires one init run plus up to 49 step runs. On the
2026-06-14 Chrome WebGPU fp16 validation run, the measured cost was dominated by
`decoder_step.onnx` execution:

| Metric | Observed |
| ------ | -------- |
| Encoder | `1759ms` |
| Decoder total | `3979ms` |
| Decoder init ORT run | `134ms` |
| Decoder step ORT run | `3788ms` across 49 steps |
| Decoder step p50 / p95 | `77ms` / `86ms` |
| JS feed build + tensor bridge + output handling | `<4ms` total |

This means a slow decoder is expected for seq2seq Whisper, and the current
profile points at ORT/WebGPU `decoder_step` graph execution rather than
JavaScript KV-cache glue. Beam search and `best_of` call `decoder_step` more
times and should be treated as quality/robustness options, not speed options.

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

### Optional precision-reference alignment

FP16 WebGPU encoder arithmetic can move an attention-DTW boundary by one
20-millisecond frame even when decoding and the alignment graph are otherwise
correct. When exact reference-quality word anchors matter, configure a paired
higher-precision alignment artifact without slowing the normal text decode:

```typescript
const source = {
  ...fastSource,
  kind: 'splitgraph' as const,
  artifacts: {
    ...fastSource.artifacts,
    alignmentReference: {
      encoderUrl: 'https://example.com/models/whisper-fp32/encoder_model.onnx',
      decoderAlignUrl: 'https://example.com/models/whisper-fp32/decoder_align.onnx',
      manifestUrl: 'https://example.com/models/whisper-fp32/manifest.json',
      externalDataUrls: {
        encoder: [{ path: './encoder_model.onnx.data', file: 'encoder_model.onnx.data' }],
        decoder_align: [{ path: './decoder_align.onnx.data', file: 'decoder_align.onnx.data' }],
      },
    },
  },
  alignmentReferenceBackend: 'webgpu',
};
```

The `alignmentReference` encoder and causal `decoderAlignUrl` must be exported
as a matched precision pair. Select it per transcription:

```typescript
await session.transcribe(audio, {
  language: 'en',
  detail: 'words',
  returnTimestamps: 'word',
  wordTimestampSource: 'reference',
});
```

`wordTimestampSource: 'fast'` always uses the primary encoder. `auto` uses the
reference pair when configured and otherwise keeps the fast path. Reference
sessions are loaded lazily only when word timestamps are requested; the native
metrics expose `wordAlignmentReferenceMs` and `wordAlignmentSource` so the
extra cost is visible.

### Alignment tensor contracts

The current exporter emits selected raw cross-attention logits with shape
`[batch, alignment_head, target_sequence, source_frames]`. The runtime crops
the fixed 30-second frame axis to the actual audio duration, applies softmax
per head, normalizes every teacher-forced row, median-filters, averages the
selected heads, and selects the rows corresponding to
`[no_timestamps, text..., last_text]`. This ordering matches Whisper's
`find_alignment` path and keeps the no-timestamps row as the leading DTW
anchor.

Older artifacts may emit an averaged post-softmax matrix with shape
`[batch, target_sequence, source_frames]`. Those graphs remain readable when
their manifest declares the legacy `post_softmax`/`mean` contract; they are
renormalized after frame cropping but cannot recover per-head short-clip
semantics.

## Verification env vars

These optional environment variables enable fixture-based tests:

| Var                              | Test                                            | What it verifies                                         |
| -------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| `WHISPER_SPLITGRAPH_FIXTURE_DIR` | `tests/whisper-splitgraph-smoke.test.ts`        | Encoder shape, init→step loop, alignment shape, row sums |
| `WHISPER_REFERENCE_JSON`         | `tests/whisper-reproducibility-harness.test.ts` | Token match vs PyTorch/ONNX Python reference             |
| `WHISPER_REFERENCE_MODEL_DIR`    | reproducibility harness                         | Override a single directory containing all graphs        |
| `WHISPER_REFERENCE_ENCODER_DIR`  | reproducibility harness                         | Override only the encoder variant directory              |
| `WHISPER_REFERENCE_DECODER_DIR`  | reproducibility harness                         | Override decoder/tokenizer/config directory              |
| `WHISPER_REFERENCE_MEL`          | reproducibility harness                         | Override the exported `.mel.npy` path                    |
| `WHISPER_REFERENCE_AUDIO`        | reproducibility harness                         | Override the WAV fixture path                            |

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

For large-v3-turbo exports that Python ONNX Runtime cannot load because its
supported ONNX IR version is older than the graph, generate the HF oracle and
mel without executing ONNX in Python:

```powershell
python tools/whisper-onnx-export/generate_hf_reference.py `
  --model-dir N:\github\asrjs\webgpu-agent-test\public\models\fp32 `
  --encoder-dir N:\github\asrjs\webgpu-agent-test\public\models\fp32 `
  --decoder-dir N:\github\asrjs\webgpu-agent-test\public\models\fp32 `
  --model-id openai/whisper-large-v3-turbo `
  --audio N:\github\asrjs\webgpu-agent-test\public\audio\jfk2.en.wav `
  --output $env:TEMP\asrjs-whisper-reference-large-v3-turbo\jfk2.reference.json `
  --export-mel --skip-onnx

$env:WHISPER_REFERENCE_JSON = "$env:TEMP\asrjs-whisper-reference-large-v3-turbo\jfk2.reference.json"
$env:WHISPER_REFERENCE_ENCODER_DIR = 'N:\github\asrjs\webgpu-agent-test\public\models\fp32'
$env:WHISPER_REFERENCE_DECODER_DIR = 'N:\github\asrjs\webgpu-agent-test\public\models\fp32'
npm test -- --run tests/whisper-reproducibility-harness.test.ts
```

The test reads input/output dimensions from graph metadata. Do not use
`manifest.max_source_positions` as the mel input-frame count: for this model it
is 1500 encoded positions, while `input_features` is `[1, 128, 3000]`.

## Architecture

```
Audio → WhisperMelProcessor → mel spectrogram
  → encoder_model.onnx → hidden_states [1, 1500, d_model]
  → decoder_init.onnx(prompt) → logits + KV cache
  → decoder_step.onnx(token, KV) × N → tokens
  → decoder_align.onnx(all_tokens) → alignment [1, N, T, 1500] logits
  → processSplitGraphAlignment() → DTW → word timestamps
```

Output is identical to merged-decoder path (`WhisperNativeTranscript`) with same
word timestamp schema.
