# Whisper 4-Graph Export & Quantization Workflow

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. EXPORT FP32 (source of truth)                        │
│    .venv/bin/python export_whisper.py MODEL_ID OUT_DIR  │
│    --device cpu --dtype float32                         │
│    Produces: encoder_model.onnx, decoder_init.onnx,     │
│              decoder_step.onnx, decoder_align.onnx,     │
│              manifest.json, tokenizer.json, ...         │
├─────────────────────────────────────────────────────────┤
│ 2. VERIFY (Python)                                      │
│    .venv/bin/python test_kv_export.py                   │
│    .venv/bin/python test_e2e_tokens.py                  │
│    .venv/bin/python test_comprehensive.py [--quantize]  │
├─────────────────────────────────────────────────────────┤
│ 3. QUANTIZE (from fp32 originals)                       │
│    Option A: --fp16 at export time                      │
│    Option B: Post-export with onnxconverter_common      │
│    Option C: --int8 at export time (dynamic quant)      │
│    All paths are external-data-safe when using          │
│    --external-data auto (or always)                     │
├─────────────────────────────────────────────────────────┤
│ 4. VERIFY (TypeScript)                                  │
│    WHISPER_SPLITGRAPH_FIXTURE_DIR=... npx vitest run    │
│    WHISPER_REFERENCE_JSON=... npx vitest run            │
├─────────────────────────────────────────────────────────┤
│ 5. UPLOAD to HF                                         │
│    gf upload MODEL_DIR REPO_ID                          │
│    hf upload REPO_ID MODEL_DIR .                        │
└─────────────────────────────────────────────────────────┘
```

## Model sizes

| Model | Params | d_model | Layers | Heads | Export device |
|-------|--------|---------|--------|-------|---------------|
| whisper-tiny | ~39M | 384 | 4 | 6 | GPU or CPU |
| whisper-base | ~74M | 512 | 6 | 8 | GPU or CPU |
| whisper-small | ~244M | 768 | 12 | 12 | GPU or CPU |
| whisper-medium | ~769M | 1024 | 24 | 16 | GPU (8GB+) |
| whisper-large-v3-turbo | ~809M | 1280 | 32 | 20 | CPU recommended |
| whisper-large-v3 | ~1.55B | 1280 | 32 | 20 | CPU only |

## Quantization plan

Workflow: `fp32 export → verify → quantize → verify → upload`

```
model-repo-root/
├── encoder_model.onnx          # fp32 (source of truth)
├── encoder_model.onnx.data     # external data (co-located)
├── decoder_init.onnx
├── decoder_init.onnx.data
├── decoder_step.onnx
├── decoder_step.onnx.data
├── decoder_align.onnx
├── decoder_align.onnx.data
├── manifest.json
├── tokenizer.json
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── README.md
├── fp16/                       # fp16 variants
│   ├── encoder_model.onnx
│   └── ...
├── int8/                       # int8 variants
│   ├── encoder_model.onnx
│   └── ...
└── manifest.json               # Updated with variant paths
```

Quantization methods:
- **fp16**: `--fp16` at export time or post-export with `convert_fp16_safe`.
  When `--external-data auto` is active, post-export fp16 uses external-data-aware
  save to stay safely below the 2 GB protobuf limit.
- **int8**: `--int8` at export time (dynamic quantization). ORT quantize_dynamic
  works on file paths and preserves external data automatically.
- **q4/q8**: Not supported by onnxconverter_common. Needs custom tooling.
- **nvfp4**: NVIDIA-specific. Not applicable for cross-platform ONNX.

## External data (large model safety)

ONNX protobuf has a 2 GB hard limit on serialized `ModelProto`.
Large Whisper models exceed this:
- whisper-large-v3-turbo (809M params): decoder_init ~910 MB, decoder_step ~606 MB
- whisper-large-v3 (1.55B params): all decoder graphs >2 GB

The exporter supports three strategies via `--external-data`:
- **auto** (default): Use external data for models with decoder_layers >= 24
- **always**: Force external data for all graphs
- **never**: Inline all weights (NOT safe for large models)

With external data enabled:
- Each `.onnx` file contains only the graph structure (small, ~400 KB).
- Weights are stored in co-located `.onnx.data` files.
- ORT loads co-located `.data` files automatically in Node.js.
- Browser loads require explicit externalData URLs in manifest + session options.

Recommended commands:
```
# CPU-safe fp32 export (avoids CUDA OOM but still needs external data)
python export_whisper.py openai/whisper-large-v3-turbo ./output \\
  --device cpu --dtype float32 --external-data auto

# GPU fp16 export (memory-efficient at load + external-data safe)
python export_whisper.py openai/whisper-large-v3-turbo ./output-fp16 \\
  --device cuda --dtype float16 --external-data auto
```

Key safety features:
- `save_onnx_safe()` — Never calls SerializeToString on >2 GB ModelProto
- `validate_onnx_safe()` — Uses path-based checker for external-data models
- `discover_external_data()` — Extracts external data metadata for manifest
- `convert_fp16_safe()` — Post-export fp16 with external-data-aware save
- `convert_int8_safe()` — ORT path-based quantize preserves external data

## Current HF repos

| Repo | Model | Variants |
|------|-------|----------|
| `ysdede/whisper-large-v3-turbo-onnx-4graph` | whisper-large-v3-turbo | fp32 |

## Next steps (quantization)

1. Re-export whisper-large-v3-turbo with `--fp16` flag → fp16 variants
2. Re-export with `--int8` flag → int8 variants
3. Organize in subdirectories: `fp16/`, `int8/`
4. Update manifest.json with variant paths
5. Upload to HF repo
6. Verify each variant with `WHISPER_SPLITGRAPH_FIXTURE_DIR`
