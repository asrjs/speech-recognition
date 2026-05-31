# Browser Model Loading — Fetch Limits, IndexedDB Cache, Pipeline API

## Problem

Browser `fetch()` fails for ONNX external data files > ~1.5-2GB:
- fp32 encoder: 2.4GB `.onnx.data` → `Failed to fetch`
- fp16/fp16io encoder: 1.2GB → works (but fp16io garbage on WASM)
- q8 encoder: 616MB inline (no `.onnx.data`) → always works

## Solutions (in order of preference)

### 1. Use q8 variant (no external data)

The q8 (int8 dynamic) variant has ALL weights inline in `.onnx` files:
- encoder: 616MB (single file, no `.data`)
- decoder_init: 228MB (single file, no `.data`)
- decoder_step: 415MB (single file, no `.data`)
- Total: ~1.3GB

Works on both WebGPU and WASM. Identical output to fp32.

### 2. Library pipeline API (for supported presets)

```javascript
import { createSpeechPipeline, buildSpeechModelLoadOptions } from '@asrjs/speech-recognition';

const pipeline = createSpeechPipeline({ cacheModels: true });

const options = buildSpeechModelLoadOptions({
  modelId: 'onnx-community/whisper-large-v3-turbo',
  backend: 'webgpu',
});

const model = await pipeline.loadModel(options);
const result = await model.transcribe(audioBuffer);
```

This handles:
- HuggingFace download with streaming
- IndexedDB caching (instant reload)
- Model loading into ORT
- Full transcription pipeline

Supported presets: `onnx-community/whisper-large-v3-turbo`, `openai/whisper-base`, etc.

### 3. Lower-level IndexedDB cache (custom models)

For custom HF repos (e.g., `ysdede/whisper-large-v3-turbo-onnx-4graph`):

```javascript
import { IndexedDbAssetCache } from '@asrjs/speech-recognition/io/cache';
import { resolveAssetHandle } from '@asrjs/speech-recognition/io/handles';

const cache = new IndexedDbAssetCache();

// resolveAssetHandle downloads from HF, caches in IndexedDB
const handle = await resolveAssetHandle({
  repoId: 'ysdede/whisper-large-v3-turbo-onnx-4graph',
  filename: 'fp32/encoder_model.onnx.data',
  subfolder: 'fp32',
}, { cache });

// Get blob URL for ORT
const url = await handle.getLocator('url');
// Or read bytes directly
const bytes = await handle.readBytes();
```

## WASM Sequential Lifecycle

When on WASM (heap limit ~1.5GB), load models sequentially:

```
1. Load encoder → run → save output → dispose
2. Load decoder_init → run → save KV → dispose
3. Load decoder_step → run loop → dispose
```

The encoder output (Float32Array) must be saved before disposal.
KV cache is passed through the session wrapper between steps.

See `webgpu-agent-test/index.html` `runSingleDecode()` for implementation.

## Model Sizes Reference

| Variant | Encoder | decoder_init | decoder_step | External data? | Works on WASM? |
|---------|---------|-------------|-------------|---------------|---------------|
| fp32 | 2.5GB | 531MB | 265MB | Yes (all) | ❌ heap limit |
| fp16 | 1.2GB | 265MB | 132MB | Yes (all) | ❌ fp16 ops |
| fp16io | 1.2GB | 531MB | 265MB | Yes (encoder) | ❌ fp16 ops |
| q8 | 616MB | 228MB | 415MB | No (inline) | ✅ |
| mixed | 616MB | 455MB | 127MB | Partial | ⚠️ |
