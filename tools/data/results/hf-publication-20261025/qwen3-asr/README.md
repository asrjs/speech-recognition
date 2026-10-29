---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - asrjs
  - qwen3-asr
  - experimental
library_name: asr.js
---

# Qwen3-ASR 0.6B — asr.js ONNX bundle

This repository contains the exact custom ONNX export and tokenizer assets used by the local `@asrjs/speech-recognition` Qwen3-ASR adapter. The upstream [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) repository is Apache-2.0 licensed and does not provide ONNX graphs at that revision. This bundle contains an FP32 audio encoder, FP16 decoder prefill/step graphs and their shared external data. It is an experimental asr.js export.

`asrjs-manifest.json` lists the exact files and SHA-256 digests. The real-weight four-demo matrix accepted 16 of 18 requested fixture flows; native streaming was unsupported. This does not certify broad multilingual accuracy, all backend/precision combinations, or physical microphones. It is not a general approved execution profile.

Use the local artifact adapter in `@asrjs/speech-recognition`; these files are not a generic Transformers.js ONNX export. Export lineage and current limitations are described in the [Qwen porting notes](https://github.com/asrjs/speech-recognition/tree/main/tools/model-debugging/reference/qwen3-asr-0.6b) and [model matrix](https://github.com/asrjs/speech-recognition/blob/main/docs/reports/HANDOFF-MODERN-MODELS-2026-10-25.md#historical-checkpoint-modern-model-e2e-matrix).
