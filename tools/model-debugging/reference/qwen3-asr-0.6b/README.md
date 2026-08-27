# Qwen3-ASR-0.6B reference tooling

This folder captures the official qwen-asr Transformers backend from a
complete local Qwen3-ASR-0.6B snapshot. The script sets Hugging Face and
Transformers offline flags before loading the package; missing files fail
locally instead of triggering a download.

## Capture a native batch reference

Install qwen-asr in an isolated environment and provide an existing local
snapshot. Official inference is `qwen_asr.Qwen3ASRModel.from_pretrained` from
PyPI `qwen-asr` (Apache-2.0). HuggingFace `Qwen/Qwen3-ASR-0.6B` has no ONNX
files; third-party graphs are not the oracle.

~~~powershell
$PYTHON = 'N:\github\asrjs\speech-recognition\tools\model-debugging\reference\qwen3-asr-0.6b\.venv\Scripts\python.exe'
$MODEL = 'N:\models\Qwen3-ASR-0.6B'

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

Official HuggingFace revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
contains no `.onnx` files. Export from `qwen-asr` 0.0.6:

- Unmodified `Qwen3ASRAudioEncoder.forward` is not ONNX-serializable
  (`aten::pad_sequence`, `.tolist()` chunk split, ragged gather).
- Static encoder path using the **same official encoder weights** (T % 100 == 0)
  matches official forward at 0 max-abs. T=800 → 104 tokens (`n_window_infer`);
  T=1100 (jfk-short) → 143 tokens.
- Thinker decoder: wrap official weights with **explicit stacked KV**
  (`present_keys`/`present_values` `[28,1,8,seq,128]`). Prefill + step graphs
  export. Native ORT greedy matches the `qwen-asr` JFK oracle. Do not treat
  `goryodog/…` or `andrewleech/qwen3-asr-onnx` as the oracle.
- WASM: sequential sessions match the JFK oracle on native fp16 (Node RSS 4254 MB)
  and fp32. `convert_float_to_float16` is `ORT_WEB_UNSUPPORTED_OP`.
- Encoder: `audio-encoder-dynamic.onnx` accepts T % 100 == 0. JS pads leftovers
  to 100 and crops tokens; T=1050 greedy matches JFK.
- Chrome WebGPU and Chrome sequential WASM (fp16 and fp32) exact JFK. Experimental, no preset.

~~~powershell
$PYTHON = 'N:\github\asrjs\speech-recognition\tools\model-debugging\reference\qwen3-asr-0.6b\.venv\Scripts\python.exe'
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/export_qwen_onnx.py `
  --model-dir N:\models\Qwen3-ASR-0.6B `
  --output-dir N:\models\onnx\qwen3-asr-0.6b-official `
  --mel-frames 1100 `
  --remainder-frames 1050 `
  --report tools/data/results/qwen/qwen3-asr-0.6b-encoder-dynamic-export.json
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/export_qwen_decoder_onnx.py `
  --model-dir N:\models\Qwen3-ASR-0.6B `
  --output-dir N:\models\onnx\qwen3-asr-0.6b-official `
  --dtype float16 `
  --report tools/data/results/qwen/qwen3-asr-0.6b-decoder-fp16-native-export.json
$env:QWEN_OFFICIAL_ONNX_SMOKE='1'
$env:NODE_OPTIONS='--max-old-space-size=16384'
npx vitest run tests/qwen3-asr-onnx-backends.test.ts
# Chrome: cd N:\github\asrjs\webgpu-agent-test
# node scripts/run-qwen-webgpu.mjs
# node scripts/run-qwen-wasm.mjs --fp16
~~~

The family stays experimental. WASM and WebGPU match on JFK. Official-graph
loads default to `audio-encoder-dynamic.onnx` (pad leftover frames to 100,
then crop tokens). Static T=1100 is opt-in (`encoder=static-t1100`). No
weights in git.
