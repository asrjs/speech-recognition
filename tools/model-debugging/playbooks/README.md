# Debugging Playbooks

These playbooks capture debugging workflows that were important enough to keep.

Use them when a bug has a repeatable shape and we want the next person to start
from a proven path instead of rediscovering the workflow.

## Current Playbooks

- [audio-prep-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\audio-prep-parity.md)
  - use when browser and Node disagree, or when resampling seems to change text quality
- [canary-aed-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\canary-aed-porting.md)
  - step-by-step NeMo AED porting flow for Canary-style models, including reference generation, ONNX export, FP16/INT8 variants, and JS frontend guidance
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)
  - the concrete `LibriVox.org` case that drove recent WAV and Node-path fixes
- [model-porting-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\model-porting-parity.md)
  - workflow for merging reference test suites and keeping CI-safe parity helpers
- [nemo-rnnt-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\nemo-rnnt-porting.md)
  - step-by-step NeMo RNNT porting flow, including frontend-contract checks for raw-log vs normalized mel features
- [gigaam-ctc-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\gigaam-ctc-porting.md)
  - official GigaAM multilingual CTC chain: checkpoint, PyTorch reference, `to_onnx`, native ORT, JS frontend
- [gigaam-rnnt-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\gigaam-rnnt-porting.md)
  - official GigaAM v3 E2E RNN-T chain: checkpoint, PyTorch reference, encoder/decoder/joint `to_onnx`, native ORT, WASM, Chrome WebGPU
- [sensevoice-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\sensevoice-porting.md)
  - official FunAudioLLM SenseVoiceSmall chain: FunASR inference, `model.export`, native ORT; OpenVoiceOS ONNX is not the oracle
- [x-asr-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\x-asr-porting.md)
  - official X-ASR-zh-en sherpa-onnx Zipformer2 streaming chain; true encoder-cache streaming
- [qwen3-asr-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\qwen3-asr-porting.md)
  - official Qwen3-ASR-0.6B chain: qwen-asr capture, static encoder, explicit KV decoder, sequential WASM, Chrome WebGPU
- [ort-webgpu-entrypoint.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\ort-webgpu-entrypoint.md)
  - keep `onnxruntime-web/webgpu` on the WebGPU bundle; diagnose provider-alias regressions before changing model code
- [huggingface-model-publishing.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\huggingface-model-publishing.md)
  - adapt a validated local model folder into our ONNX repo card format and publish it with `hf` CLI

## When To Create A New Playbook

Create one when:

- the bug took multiple comparison steps to isolate
- more than one script was needed
- the investigation produced a reusable rule
- future model-family work is likely to hit the same class of issue

Good playbooks usually include:

- the symptom
- the environment assumptions
- the exact fixtures used
- the scripts to run
- the expected outputs
- the conclusion
- any library fixes that resulted from the investigation
