# GigaAM RNN-T encoder fp16 quantization probe (2026-08-30)

## Setup

The official GigaAM v3 E2E RNN-T encoder ships fp32-only at 844 MB. A
local fp16 variant was produced with
onnxruntime.transformers.float16.convert_float_to_float16
(keep_io_types=True, disable_shape_infer=True) via the reproducible tool
tools/scripts/convert_gigaam_rnnt_encoder_fp16.py. Output:
N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.fp16.onnx (442.8 MB).
The float32 graph contract is preserved: casts are internal, inputs and
outputs stay float32.

## Numerical parity (ORT CPU, real example.wav features)

Features came from the library's own GigaAmJsPreprocessor on the official
11.29 s example.wav (1128 frames x 64 mels, mel-major [1, 64, 1128]):

- encoder output shape: [1, 768, 282] float32 on both graphs
- max abs diff 2.305e-03, mean abs diff 2.803e-04, cosine 0.9999987

## Browser end-to-end (Chrome headless, NVIDIA Blackwell, enc WebGPU +
dec/joint WASM, 1 warm-up + 3 runs, exact Russian oracle)

| Encoder | Median ms | RTFx | Encode ms | Load ms | Parity |
|---|---|---|---|---|---|
| fp32 (default) | 392.9 | 28.7x | 70.1 | 7999 | exact |
| fp16 | 412.8 | 27.4x | 130.7 | 5258 | exact |

## Verdict

- Valid as a size / load / VRAM option: 844 -> 443 MB (47% smaller), load
  8.0 -> 5.3 s (~34% faster), exact transcript parity, CPU numerical
  agreement cosine 0.9999987.
- Not valid as a speed option: the fp16 graph encodes ~2x slower on
  WebGPU here (internal casts from keep_io_types plus fp16 kernel
  scheduling), costing ~5% end-to-end RTFx.
- Keep fp32 as the default; document fp16 as the load/size alternative.
  This confirms the playbook rule: quantization decisions are measured,
  never assumed.

## Harness changes

The GigaAM RNN-T browser runner accepts --encoder-file=NAME and the page
accepts encoderFile=NAME to load alternate encoder artifacts from the
same served folder.

## Artifacts and reproduction

python tools/scripts/convert_gigaam_rnnt_encoder_fp16.py
node scripts/run-gigaam-rnnt-webgpu.mjs --encoder-file=v3_e2e_rnnt_encoder.fp16.onnx --warmup=1 --repeat=3
tools/data/results/gigaam/v3-rnnt-fp16-enc-gpu-decwasm-librivox.json

