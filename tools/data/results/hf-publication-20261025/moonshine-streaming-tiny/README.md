---
license: mit
pipeline_tag: automatic-speech-recognition
tags:
  - onnx
  - asrjs
  - moonshine
  - streaming
  - experimental
library_name: asr.js
---

# Moonshine Streaming Tiny — asr.js ONNX bundle

A custom remapped four-graph FP32 bundle for the streaming Tiny model, using the encoder, chunked encoder, decoder prefill and decoder step graphs plus external data and processor files. It is for the asr.js Moonshine streaming adapter; it is distinct from offline Moonshine Tiny.

Source model: [`moonshine-ai/moonshine-streaming-tiny`](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny/tree/f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a), revision `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`. The source model repository identifies its streaming models as MIT licensed. The included [`LICENSE`](LICENSE) contains that MIT grant and attribution. `asrjs-provenance.json` records the export graph hashes without local machine paths.

## Validation scope

The model has a partial real-weight app matrix: 13 of 18 tested fixture flows passed. Manual capture and Playground recording were empty on padded input; WebGPU FireRed VAD missed one word. These results do not certify physical microphones, broad accuracy, or every backend combination. The graphs are marked experimental; see [`asrjs-manifest.json`](asrjs-manifest.json) for each included file's hash and exact scope.

Use the asr.js local artifact adapter. This export's graph mapping and external data layout are specific to that runtime.
