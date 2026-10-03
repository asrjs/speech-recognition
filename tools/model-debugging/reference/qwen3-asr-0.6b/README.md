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

```powershell
$PYTHON = 'N:\github\asrjs\speech-recognition\tools\model-debugging\reference\qwen3-asr-0.6b\.venv\Scripts\python.exe'
$MODEL = 'N:\models\Qwen3-ASR-0.6B'

& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/capture_qwen_reference.py `
  --model-dir $MODEL `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --output tools/data/results/qwen/qwen3-asr-0.6b-reference.json `
  --device-map cpu `
  --dtype float32 `
  --batch-size 1 `
  --max-inference-batch-size 1 `
  --max-new-tokens 256
```

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

```powershell
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
```

The family stays experimental. WASM and WebGPU match on JFK. Official-graph
loads default to `audio-encoder-dynamic.onnx` (pad leftover frames to 100,
then crop tokens). Static T=1100 is opt-in (`encoder=static-t1100`). No
weights in git.

## Reuse the existing offline library runner

The `qwen-asr` family already exists under `src/models/qwen-asr`. Do not write
another decoder loop to exercise local artifacts. Use the existing
[Node/WASM benchmark CLI](../../../../tests/smoke/qwen3-asr-node-wasm-benchmark.mjs).

Check that the local graphs, external-data shards and tokenizer are present.
The historical original-snapshot path above is not proof that the PyTorch
weights still exist. ONNX output can be exercised without that snapshot; a
fresh official-engine capture requires the original weights and reference
environment. Neither an old receipt nor `--help` proves current inference.

From the repository root, after building the current library:

```powershell
node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --help
node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs `
  --model-dir N:\models\onnx\qwen3-asr-0.6b-official `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --backend wasm --encoder dynamic --dtype fp16 `
  --warmup 0 --runs 1 --language English `
  --reference tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-reference.json
```

This command uses local artifacts and the library's executor. It does not
download weights or establish feature/encoder/cache numerical parity.
Use the [stage comparator](../../scripts/README.md#stage-capture-contract)
for recorded stage evidence.

## Shared encoder-boundary control

Reuse [`tools/stage_attribution.py`](../../../stage_attribution.py) through a
callback that runs the existing encoder, not through another decoding loop.
Keep preparation model-owned: `Qwen3AsrFeatureProcessor` produces mel-major
float32 `input_features` shaped `[128,T]`. The dynamic graph returns
`audio_embeddings` shaped `[audio_tokens,1024]`.

A real CPU encoder exercise is recorded in
`tools/data/results/qwen/goal-harness-encoder-layout-attribution-jfk-2026-10-03.json`.
The receipt contains the tested Node preparation and six-line Python stage
callback. Its prepared JFK input is `[128,1100]`, bound to PCM SHA-256
`ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
Seven source/artifact hashes were unchanged across execution.

This is a **controlled layout-negative**, not a production frontend finding:
the reference is current library preparation, not original-engine features.
The candidate uses `np.ascontiguousarray(reference_features.T).reshape(reference_features.shape)`,
preserving shape/dtype while corrupting layout. One
`attribute_input_source(..., atol=1e-5, rtol=1e-4)` invocation performs three
actual encoder calls, attributes the failure to `input_features`, and reproduces
the candidate output with zero numerical error. This one-input case does not
demonstrate multi-input discrimination.

The existing library CLI also produced saved-reference-matching JFK output in
`goal-harness-existing-port-jfk-2026-10-03.json`: WER/CER zero, 25 tokens,
dynamic encoder and fp16 decoder on WASM. Cold execution took 111.9 seconds;
do not treat this as performance, browser or fresh official-engine numerical
acceptance. See the [measured reuse scope](../../../../docs/GOAL_MODEL_PORT_HARNESS.md#existing-qwen-port-real-output-and-cross-family-control).

## Optional decoder graph surgery

`append_argmax_output.py` creates a copy of an explicit-KV decoder graph with
an `INT64 next_token_id` output computed by ONNX `ArgMax`. Use
`--remove-logits` only for the candidate measurement; the default retains
`logits` as a control. The tool loads with external data disabled and saves
without rewriting the referenced shard, then checks the serialized candidate.

```powershell
$PYTHON = 'N:\github\asrjs\speech-recognition\tools\model-debugging\reference\qwen3-asr-0.6b\.venv\Scripts\python.exe'
$DIR = 'N:\models\onnx\qwen3-asr-0.6b-official'
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/append_argmax_output.py `
  --input "$DIR\decoder-prefill.onnx" `
  --output "$DIR\decoder-prefill-argmax.onnx" `
  --remove-logits
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/append_argmax_output.py `
  --input "$DIR\decoder-step.onnx" `
  --output "$DIR\decoder-step-argmax.onnx" `
  --remove-logits
```

This is not automatically faster. The 2026-08-29 Chrome/ORT Web 1.29 probe
preserved exact JFK text but lowered warmed RTFx from 6.7778x to 4.1992x as
the provider's reduction work outweighed the cheaper output handling. Keep
the original graphs as the production control until a provider-specific
fusion/reduction improvement is measured.
