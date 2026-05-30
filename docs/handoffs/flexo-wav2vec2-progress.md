# Flexo — WAV2VEC2 CTC Model Progress

Branch: `feat/large-v3-turbo-fp16-external-data` (also exists on `feat/asr-pipeline-output-formats`)
Last updated: 2026-05-30
Author: Flexo (deepseek-v4-pro, home P520)

## Current Status

**COMPLETE.** Two Wav2Vec2 models (EN + TR) ported to ONNX, published on HuggingFace with 3 quantization variants each (fp32, fp16, q8). ASR + WhisperX-style forced alignment validated on Node/WASM and native ORT. WebGPU smoke HTML ready.

Completed:
- `src/ctc/` shared CTC module
- `src/models/wav2vec2/` model family (types, config, tokenizer, ORT, executor, model, mapping)
- `src/presets/wav2vec2/` presets for 6 model variants (EN fp32/fp16/q8, TR fp32/fp16/q8)
- `src/alignment/` CTC Viterbi + Wav2Vec2 aligner + extractLogits + createWav2Vec2AlignerFromLogits
- Built-in runtime registration with `useManifestSource: true`
- Node.js HF download bridge (materialize to temp for ORT)
- Quantization benchmark suite (native ORT, Python + JS)

## HF Repos

| Repo | Models | Variants |
|------|--------|----------|
| `ysdede/wav2vec2-base-960h-onnx` | EN base-960h (95M) | fp32, fp16, q8 |
| `ysdede/wav2vec2-large-xlsr-turkish-onnx` | TR large-xlsr (317M) | fp32, fp16, q8 |

## Quantization Benchmarks (native ORT, CPU, P520)

### English — `facebook/wav2vec2-base-960h` (WhisperX default, JFK 11s)

| Variant | Size | Infer | WER | Preset Alias |
|---------|------|-------|-----|-------------|
| fp32 | 362 MB | 704ms | 4.5% | `facebook/wav2vec2-base-960h` |
| **fp16** | **182 MB** | **769ms** | **4.5%** | `base-960h-fp16` |
| q8 | 91 MB | 2291ms | 9.1% | `base-960h-q8` |

### Turkish — `m3hrdadfi/wav2vec2-large-xlsr-turkish` (WhisperX default, 18.6s)

| Variant | Size | Infer | WER | Preset Alias |
|---------|------|-------|-----|-------------|
| fp32 | 1207 MB | 5626ms | 53.6% | `wav2vec2-turkish` |
| **fp16** | **605 MB** | **3856ms** | **53.6%** | `wav2vec2-turkish-fp16` |
| q8 | 302 MB | 5888ms | 71.4% | `wav2vec2-turkish-q8` |

**Conclusion: fp16 is optimal for both models.**
- Same WER as fp32 (identical transcript output)
- 2x smaller download
- For large models (317M+): also 32% faster inference
- q8 degrades accuracy (WER +4-18 points) and is slower (int8→fp32 dequant overhead on CPU)
- Use fp16 as default, fp32 for debugging, q8 only for size-constrained deployment

## Export Recipes

**fp16** (PyTorch export-time):
```python
model = model.half()  # convert to float16
dummy = torch.randn(1, 16000*10, dtype=torch.float16)
torch.onnx.export(model, dummy, path, opset_version=18, do_constant_folding=True,
    input_names=["input_values"], output_names=["logits"],
    dynamic_axes={"input_values":{0:"batch",1:"sequence"},"logits":{0:"batch",1:"frames"}})
```

**q8** (post-export, requires optimization pass first):
```python
from onnxruntime.transformers import optimizer, FusionOptions
from onnxruntime.quantization import quantize_dynamic, QuantType

opt = optimizer.optimize_model(src, 'bert', num_heads=16, hidden_size=1024, ...)
opt.save_model_to_file(opt_path)
quantize_dynamic(opt_path, dst, weight_type=QuantType.QInt8)
```

**Pitfall**: Direct `quantize_dynamic` without `optimize_model` fails with `ValueError: Expected mul_N to be an initializer`. Torch export with `do_constant_folding=True` doesn't fold all patterns — the optimizer handles the remaining non-initializer Conv weights.

**opt-fp32**: ORT optimizer converts external data to inline, producing 1.2 GB+ single files that exceed WASM heap limit (OOM). Not recommended.

## ORT Backend Notes

- **Local smoke tests**: Use native ORT (`onnxruntime` Python / `onnxruntime-node`), NOT WASM
- **WASM**: ~1.5 GB heap limit, single-session constraint. Large models (1.2 GB) fail on WASM.
- **Native ORT**: Full system RAM, persistent multi-session support. Reference backend for benchmarking.
- **Benchmark script**: `tests/smoke/wav2vec2-tr-quant-bench.mjs` (JS, direct session) and Python equivalent

## All Preset Aliases

| Model | Alias | Variant |
|-------|-------|---------|
| EN | `facebook/wav2vec2-base-960h`, `wav2vec2`, `base-960h` | fp32 |
| EN | `base-960h-fp16`, `wav2vec2-fp16` | fp16 |
| EN | `base-960h-q8`, `wav2vec2-q8` | q8 |
| TR | `wav2vec2-turkish`, `xlsr-turkish`, `wav2vec2-tr` | fp32 |
| TR | `wav2vec2-turkish-fp16`, `xlsr-turkish-fp16` | fp16 |
| TR | `wav2vec2-turkish-q8`, `xlsr-turkish-q8` | q8 |

## Usage

```typescript
import { loadSpeechModel } from '@asrjs/speech-recognition';
import { createWav2Vec2AlignerFromLogits } from '@asrjs/speech-recognition/alignment';

// ASR — fp16 (recommended)
const model = await loadSpeechModel('base-960h-fp16', { useManifestSources: true });
const result = await model.transcribe(audio);

// Forced alignment (WhisperX-style)
const logits = await model.session.executor.extractLogits(audio);
const aligner = createWav2Vec2AlignerFromLogits(logits);
const words = aligner.align({ transcript: 'known text' });

// Turkish
const trModel = await loadSpeechModel('wav2vec2-turkish-fp16', { useManifestSources: true });
```

## Key Design Decisions

- fp16 is default recommendation (not q8 like many LLMs). Unlike LLMs where q8 preserves accuracy, Wav2Vec2 CTC suffers significant WER degradation from q8 (+4-18 points) because CTC argmax is sensitive to logit precision.
- `modelDataFilename` defaults removed — must be explicitly set for models with external data. q8 models (inline weights) skip this field entirely.
- Conv bias=true for XLSR models (Turkish), false for base (English).
- feat_extract_norm: 'layer' for XLSR, 'group' for base.
