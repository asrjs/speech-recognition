# Qwen3-ASR-0.6B reference tooling

This folder captures the official qwen-asr Transformers backend from a
complete local Qwen3-ASR-0.6B snapshot. The script sets Hugging Face and
Transformers offline flags before loading the package; missing files fail
locally instead of triggering a download.

## Capture a native batch reference

Install qwen-asr in an isolated environment and provide an existing local
snapshot:

~~~powershell
$PYTHON = 'C:\path\to\qwen3-asr\python.exe'
$MODEL = 'N:\models\Qwen3-ASR-0.6B'

& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/capture_qwen_reference.py ~
  --model-dir $MODEL ~
  --audio tools/data/fixtures/audio/jfk-short.wav ~
  --output tools/data/results/qwen/qwen3-asr-0.6b-reference.json ~
  --device-map cpu ~
  --dtype float32 ~
  --batch-size 1 ~
  --max-inference-batch-size 1 ~
  --max-new-tokens 256
~~~

For a fixed language, add --language English (or another model-supported
language). To capture timestamps, pass a complete local
Qwen3-ForcedAligner-0.6B snapshot with --forced-aligner-dir and add
--timestamps.

The JSON preserves detected language, text, optional timestamps, audio
identity, batch order, runtime settings, and the raw native result object.
Use --hash-model-files when the reference is important enough to pay the
cost of hashing the entire snapshot.

## ONNX boundary

Qwen3-ASR is an audio-conditioned language model, not a Whisper split graph.
Before writing an ONNX exporter or WebGPU runtime, capture this reference,
inspect the package frontend and multimodal packing, and freeze the
generation/cache contract. Then audit and compare the smallest exported graph
with node-audit-onnx-artifact.mjs; native ORT and WASM parity must precede
WebGPU work.
