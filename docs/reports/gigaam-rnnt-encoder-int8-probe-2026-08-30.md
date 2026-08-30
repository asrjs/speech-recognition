# GigaAM RNN-T encoder INT8 probe (2026-08-30)

## Scope

This is the untried INT8 follow-up to the GigaAM v3 RNN-T fp16 encoder probe.
The source encoder remains untouched:

- fp32 source: `N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.onnx`
- source size: 844.1 MB
- captured official features: `N:/models/gigaam/v3-e2e-rnnt/captures/example.features.npy`
  (shape `[1,64,1128]`, SHA-256
  `d3701379f5717c1d039fee8d039e56c333aa44f74fbd22df8356558c83f3525e`)

The reproducible converter is
`tools/scripts/convert_gigaam_rnnt_encoder_int8.py`. It uses
`onnxruntime.quantization.quantize_dynamic`, per-channel QInt8 weights, and
keeps the float32 input/output contract.

## Candidate variants

### MatMul + Conv (rejected at runtime)

The initial broad probe produced a 225.4 MB graph containing
`MatMulInteger` and `ConvInteger`. ORT CPU failed session creation with:

`NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10)`.

This graph is not retained as a library artifact and is not a viable
cross-provider candidate. Reproduce the negative probe with:

`python tools/scripts/convert_gigaam_rnnt_encoder_int8.py --per-channel`

### MatMul-only (measured candidate)

Conversion:

`python tools/scripts/convert_gigaam_rnnt_encoder_int8.py --ops MatMul
--per-channel --output
N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.int8-matmul.onnx`

Output size: **320.1 MB** (62.1% smaller than fp32; 27.7% smaller than the
442.8 MB fp16 probe).

## Numerical and recognition correctness

The ORT CPU parity harness
`tools/scripts/check_gigaam_rnnt_encoder_int8_parity.py` compares the
captured official feature tensor against fp32 and writes
`tools/data/results/gigaam/v3-rnnt-encoder-int8-matmul-parity.json`.

- output shape: `[1,768,282]` on both graphs
- output length: `282` on both graphs
- max abs diff: **0.471812**
- mean abs diff: **0.046661**
- cosine similarity: **0.992093**
- CPU run: 427 ms fp32 reference, 352 ms INT8 candidate

The encoder numerical drift is material, so transcript parity is the
promotion gate. Both real-artifact browser runs produced the exact fixed
Russian oracle transcript.

## Browser measurements

Harness: `N:/github/asrjs/webgpu-agent-test/scripts/run-gigaam-rnnt-webgpu.mjs`,
11.29 s `example.wav`, one warm-up plus three measured runs, exact oracle,
NVIDIA Blackwell, ORT Web 1.29.0.

| encoder/provider | median transcribe | RTFx | oracle |
| --- | ---: | ---: | --- |
| fp32 WebGPU (recorded control) | 449.3 ms | 25.13x | exact |
| INT8 MatMul WebGPU | 2676.1 ms | 4.22x | exact |
| fp32 WASM | 5059.9 ms | 2.23x | exact |
| INT8 MatMul WASM | 3466.3 ms | 3.26x | exact |

Individual INT8 WebGPU runs were 2606.0, 2676.1, and 2801.1 ms
(RTFx 4.33, 4.22, 4.03). The graph executes on WebGPU, but the
`DynamicQuantizeLinear + MatMulInteger` path is about **6x slower** than the
fp32 WebGPU encoder. INT8 is also slower than the fp32 WebGPU hybrid despite
the 62% size reduction. On CPU/WASM it improves the fp32-WASM control by about
1.46x, but remains far behind the hybrid GPU path.

Evidence JSONs:

- `tools/data/results/gigaam/v3-rnnt-encoder-int8-matmul-parity.json`
- `tools/data/results/gigaam/v3-rnnt-int8-matmul-encgpu-decwasm-example.json`
- `tools/data/results/gigaam/v3-rnnt-int8-matmul-encwasm-decwasm-example.json`
- fp32 control: `tools/data/results/gigaam/v3-rnnt-encgpu-decwasm-1t-librivox.json`

## Decision

**Do not promote INT8 MatMul to the WebGPU default.** Keep fp32 as the
performance default and retain INT8 as an experimental CPU fallback / model
size option, subject to application accuracy tolerance. The measured
cosine/drift and exact-oracle result should both be preserved: an aggregate
transcript pass does not erase encoder numerical error.

This closes the GigaAM RNN-T encoder INT8 question for the current ORT Web
1.29/NVIDIA Blackwell stack. A future static INT8 or weight-only graph is a
separate hypothesis and must repeat the same parity and browser gates; do not
assume it will inherit this result.

