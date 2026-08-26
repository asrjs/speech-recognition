# Parakeet TDT v3 local baseline

Date: 2026-08-26
Workspace: `N:\github\asrjs\speech-recognition`

This report records a local, reproducible baseline for the already integrated
Parakeet TDT v3 preset. It is evidence for the working baseline, not a claim
that every precision/provider combination is promoted.

## Artifact provenance

- Original model: `nvidia/parakeet-tdt-0.6b-v3`
- ONNX conversion baseline: `istupakov/parakeet-tdt-0.6b-v3-onnx`
- Conversion snapshot: `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`
- Local artifact: `N:\models\onnx\nemo\parakeet-tdt-0.6b-v3-onnx`
- License recorded by the artifact: CC-BY-4.0
- Audit output: `provenance\asrjs-audit-20260826.json`

The local audit found seven ONNX graphs and loaded all seven through native
CPU ONNX Runtime without a graph-load failure:

| Component             | Variants         | Boundary                                                         |
| --------------------- | ---------------- | ---------------------------------------------------------------- |
| `encoder-model`       | fp32, fp16, int8 | `audio_signal`, `length` → `outputs`, `encoded_lengths`          |
| `decoder_joint-model` | fp32, fp16, int8 | `encoder_outputs`, targets, predictor states → logits and states |
| `nemo128`             | one              | waveform and length → 128-bin features and feature length        |

The fp32 encoder uses co-located external data. The audited sidecar is
2,435,420,160 bytes with SHA-256
`9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36`.

### Operator inventory

The optional Python ONNX inspection completed for all seven graphs. Every
graph uses ONNX opset 17. The fp32/fp16 encoders contain 4,491/4,493 nodes;
the int8 encoder contains 5,654 nodes and includes dynamic quantization and
integer matrix multiplication. The int8 decoder imports the `com.microsoft`
domain, while `nemo128.onnx` contains the custom `this:nemo_preprocessor`
operator domain.

These are provider-risk hints, not failures. The measured int8 WASM run below
proves this exact composition works there; WebGPU still requires its own exact
artifact/provider run before promotion.

## Library run

Command:

```powershell
node tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs `
  --model-id parakeet-tdt-0.6b-v3 `
  --model-dir N:\models\onnx\nemo\parakeet-tdt-0.6b-v3-onnx `
  --audio tools\data\fixtures\audio\librivox.org.wav `
  --encoder-quant int8 `
  --decoder-quant int8 `
  --output N:\models\onnx\nemo\parakeet-tdt-0.6b-v3-onnx\provenance\asrjs-v3-inspect-int8-20260826.json
```

Observed result:

| Metric             |                                    Value |
| ------------------ | ---------------------------------------: |
| Audio              | 18.714 s, 22.05 kHz source → 16 kHz mono |
| Backend            |                                     WASM |
| Load               |                                 2,512 ms |
| Transcription      |                                 4,068 ms |
| RTFx               |                                  4.6082× |
| Tokens / words     |                                  91 / 41 |
| Decoder iterations |                                      104 |
| Warnings           |                                     none |

The transcript was:

> Preface of A Year with the Birds. This is a LibriVox recording. All LibriVox recordings are in the public domain. For more information, or to volunteer, please visit LibriVox.org. Read by Olivia. A Year with the Birds by W. Ward Fowler.

The complete machine-readable output remains beside the local artifact at
`provenance\asrjs-v3-inspect-int8-20260826.json`.

## Boundary and next comparison

This run proves the current library path, local asset resolution, int8 WASM
execution, canonical transcript shaping, timing metadata, and cleanup path for
one fixed English clip. It does not prove WebGPU parity, Turkish quality, or
cross-variant numerical parity.

The reusable stage comparator is now available at
`tools/model-debugging/scripts/node-compare-stage-captures.mjs`. A future
reference capture should compare Parakeet, Qwen3, or X-ASR by stable
`sample_id` and audio identity before comparing quality scores.
