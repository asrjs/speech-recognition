# Four-family experimental ASR handoff

Date: 2026-08-27 (updated same day with GigaAM RNN-T + HUD wiring)
Workspace: `N:\github\asrjs\speech-recognition`

No presets. Weights stay under `N:\models` (not git). Chrome WebGPU is the
supported browser path where measured; Qwen also has sequential WASM.

Do **not** mix GigaAM RNN-T (Russian `example.wav`) with the four JFK CTC /
speech-LLM claims below.

Oracle JFK sentence (four families only: GigaAM CTC, SenseVoiceSmall,
X-ASR zh-en 160ms, Qwen3-ASR 0.6B):

> And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.

## Provenance

| Family            | Official source                                                                   | Oracle                                                      |
| ----------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| GigaAM CTC        | GigaAM `multilingual_ctc` + `to_onnx`                                             | official PyTorch / native ORT                               |
| SenseVoiceSmall   | FunAudioLLM `model.export`                                                        | FunASR, not OpenVoiceOS ONNX                                |
| X-ASR zh-en 160ms | sherpa-onnx Zipformer2 streaming                                                  | sherpa; JS fbank = knf `snip_edges=false`, `high_freq=-400` |
| Qwen3-ASR 0.6B    | `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, `qwen-asr==0.0.6` | official CPU; third-party ONNX is not the oracle            |

Qwen ONNX lives at `N:\models\onnx\qwen3-asr-0.6b-official\`. Unmodified encoder
`aten::pad_sequence` remains `EXPORT_BLOCKED`. Decoder is explicit stacked KV
(prefill + step), not HuggingFace `DynamicCache`.

## Graphs

### Qwen encoder

- Static T=1100: `audio-encoder-static-t1100.onnx` (747,177,039 bytes, SHA-256 `37b562fe9f4eb207a3ed5f3ec41938e528e503be29a8f9d2fe71f103d7707444`).
- Dynamic: `audio-encoder-dynamic.onnx` (751,103,768 bytes, SHA-256 `7a7f72361b1809a03e2f60b058a49018cf5e788ff07d979f9d554c0706ba42e6`). Input `[128, T]` with `T % 100 == 0`. Native ORT vs PyTorch ~7e-7 at T=800 and T=1100.
- Pad-to-chunk: JS pads leftover frames to the next 100, then crops tokens with the official `_get_feat_extract_output_lengths` formula. On T=1050, embeddings differ from the official ragged last chunk (max-abs 0.054) but greedy text still matches the JFK oracle.

### Qwen decoder

- fp32 prefill/step: ~3.006 GB `.onnx.data` each (byte-identical weights).
- Native fp16 (`torch.float16` export, **not** `convert_float_to_float16`): shared `decoder-fp16.onnx.data` 1,503,250,432 bytes, SHA-256 `a37bc27a1abe435dcd22b1637b4244650c333a591391c342d86aaac1fd0fd675`. I/O is float16. Native ORT greedy exact JFK.
- `convert_float_to_float16` remains `ORT_WEB_UNSUPPORTED_OP` (SimplifiedLayerNormFusion + inserted Casts). Do not use it for WASM.

Sequential session load (encoder → release → prefill → release → step) is required so WASM can allocate. Encoder + decoder together is still `WASM_MEMORY_LIMIT`.

## Failure classes

- `EXPORT_BLOCKED` — unmodified Qwen encoder `pad_sequence` / decoder `DynamicCache`
- `WASM_MEMORY_LIMIT` — encoder and 3 GB decoder loaded at once
- `ORT_WEB_UNSUPPORTED_OP` — `convert_float_to_float16` graphs on onnxruntime-web
- `WEBGPU_NO_ADAPTER` — Node/vitest has no GPU
- `PREPROCESSING_MISMATCH` — reserved for text divergence vs oracle

## Browser results (JFK)

| Family            | Chrome WebGPU                                              | WASM                                                                                                                                           |
| ----------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| GigaAM CTC        | exact, official fp16                                       | exact (fp32 and official fp16)                                                                                                                 |
| SenseVoiceSmall   | exact                                                      | exact                                                                                                                                          |
| X-ASR zh-en 160ms | exact                                                      | exact (same as sherpa, no extra comma)                                                                                                         |
| Qwen3-ASR 0.6B    | exact; GPU-KV median 4.86 s over three loaded-session runs | sequential fp16 and fp32 exact; Chrome fp16 JS heap 2082/4192 MB; Chrome fp32 also passed (WASM linear memory is outside `performance.memory`) |

Fresh Qwen Node WASM fp16: RSS 4317 MB, 33.15 s, RTFx 0.317. Chrome WASM
fp16: 46.5 s, RTFx 0.24. WebGPU remains the faster browser path.

### GigaAM CTC phase profile (2026-08-29)

The CTC executor now exposes distinct preprocessing, ORT encoder, and
readback/decode metrics. Three fresh Chrome/NVIDIA Blackwell runs of the
official fp16 graph had a warm median of `363.755 ms` (`31.8025x` RTFx):
`74.295 ms` preprocessing (20.42%), `283.050 ms` encoder (77.81%), and
`2.820 ms` CTC decode/readback (0.78%), with exact JFK parity in all runs.
The measured bottleneck is the encoder, so CTC tensor-copy/argmax surgery is
deferred until a graph/provider experiment can move that larger phase.
Evidence: `docs/reports/gigaam-ctc-webgpu-phase-profile-2026-08-29.json`.

## Promotion criteria (do not invent presets)

Keep all four **experimental**. A public preset needs:

1. Oracle-matching greedy text on the checked fixture (done for JFK).
2. At least one supported browser path (Chrome WebGPU is that path for all four; Qwen sequential WASM is an extra fallback, not a reason to drop WebGPU).
3. Qwen additionally needs the dynamic encoder in the default load path (done: library `resolveOfficialQwen3AsrDirectArtifacts` and Chrome/Qwen harness default to `audio-encoder-dynamic.onnx`; static T=1100 is opt-in).
4. No third-party ONNX as the Qwen oracle.
5. Streaming/long-audio contracts remain family-specific (X-ASR is true encoder-cache streaming; Qwen is a within-model-limit offline speech-LLM, while generic composition beyond 30 seconds remains compatibility-only until broader oracle coverage exists).

## Commands

```powershell
$PYTHON = 'N:\github\asrjs\speech-recognition\tools\model-debugging\reference\qwen3-asr-0.6b\.venv\Scripts\python.exe'
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/export_qwen_onnx.py --mel-frames 1100 --remainder-frames 1050 --report tools/data/results/qwen/qwen3-asr-0.6b-encoder-dynamic-export.json
& $PYTHON tools/model-debugging/reference/qwen3-asr-0.6b/export_qwen_decoder_onnx.py --dtype float16 --report tools/data/results/qwen/qwen3-asr-0.6b-decoder-fp16-native-export.json
$env:QWEN_OFFICIAL_ONNX_SMOKE='1'; $env:NODE_OPTIONS='--max-old-space-size=16384'
npx vitest run tests/qwen3-asr-onnx-backends.test.ts
# Chrome, Vite :8765 already running:
# cd N:\github\asrjs\webgpu-agent-test
# node scripts/run-qwen-wasm.mjs --fp16
# node scripts/run-qwen-wasm.mjs
```

---

## GigaAM RNN-T `v3_e2e_rnnt` (Russian-only, experimental)

This is a **fifth experimental family**, not a JFK gate and not a public preset.
Discover via `listExperimentalSpeechFamilies()` / `getExperimentalSpeechFamily('gigaam-rnnt')`.
`getSpeechModelDescriptor('gigaam-v3-e2e-rnnt')` is null. `listSpeechModels()` stays preset-only.

### Oracle (do not mix with JFK)

Official GigaAM `example.wav` (Pushkin), 11.29s, SHA-256
`d8aaaa18a5098d7c6de0595ae7ac1e64cacd0d4022af3595213bdaf23be77e69`:

> Ничьих не требуя похвал, Счастлив уж я надеждой сладкой, Что дева с трепетом любви Посмотрит, может быть, украдкой На песни грешные мои. У лукоморья дуб зелёный.

### Provenance

- Official repo: `https://github.com/salute-developers/GigaAM` @ `7447938d791c4f3e643386ee22c33777004293a5`
- Checkpoint: `N:\models\gigaam\official-cache\v3_e2e_rnnt.ckpt` (448,929,252 bytes)
  - MD5 `2730de7545ac43ad256485a462b0a27a`
  - SHA-256 `f60c62fe45902d967000770a12f260de7c5ac4d4b8e5e852f1df8c548e3958b5`
- Tokenizer: official `.model` in the same cache; JS uses dumped pieces
  `v3_e2e_rnnt_vocab.txt` (blank **1024**, vocab 1024, `num_classes` 1025)
- ONNX dir: `N:\models\onnx\gigaam\v3-e2e-rnnt` (`model.to_onnx`, opset 17, float32)
- Export status in `provenance.json`: `experimental-official-export`

| Artifact                   | Bytes       | SHA-256                                                            |
| -------------------------- | ----------- | ------------------------------------------------------------------ |
| `v3_e2e_rnnt_encoder.onnx` | 885,093,282 | `41ef815cebb6cdc7158321ec2a8b4d1ab04d1eb55f1c36ff98b428d39d0866a7` |
| `v3_e2e_rnnt_decoder.onnx` | 4,599,970   | `fa95ac0997e621ebee4156fc9295e57407549e2d5c20ed548626ea1e4f20a09c` |
| `v3_e2e_rnnt_joint.onnx`   | 2,712,926   | `2ead0f16a18554b8d557875110da4b1e964441f617cdd95d0b4b45ac999c60a9` |
| `v3_e2e_rnnt_vocab.txt`    | 14,379      | `b98730cffb0bb782f505003caff8b4ba03ef6c9baf6b799572af3191f42fe098` |

Graph I/O:

- encoder `audio_signal` / `length` → `encoded` / `encoded_len`
- decoder `x` / `hi` / `ci` → `dec` / `ho` / `co`
- joint `enc` / `dec` → `joint`

Official greedy: blank does **not** update predictor LSTM state. `hi`/`ci` are `[layers, batch, hidden]`.
Mixed WebGPU/WASM sessions must load **sequentially**: ORT can initialize WASM
fallback kernels from a WebGPU session, and concurrent creation reproduces
`multiple calls to 'initWasm()' detected`. An opt-in all-WebGPU startup probe
(`parallelSessionInitialization: true`) overlaps the three independent graph
loads; keep it experimental because earlier ORT builds also reported
`another WebGPU EP inference session is being created` for `Promise.all`.

### Gates (exact greedy text vs official `example.wav`)

| Gate                                   | Result                                        |
| -------------------------------------- | --------------------------------------------- |
| PyTorch `transcribe`                   | exact                                         |
| Native ORT greedy                      | exact; encoder max-abs ~3e-6 vs PyTorch       |
| Node WASM (`GIGAAM_RNNT_ONNX_SMOKE=1`) | exact                                         |
| Chrome WebGPU (NVIDIA Blackwell)       | exact; load ~8.0s, transcribe ~5.3s, RTF 0.47 |
| Node WebGPU                            | `WEBGPU_NO_ADAPTER`                           |

Chrome result: `tools/data/results/gigaam/v3-e2e-rnnt-example-webgpu-chrome.json`
Harness: `N:\github\asrjs\webgpu-agent-test` — `gigaam-rnnt.html`, asset route `/gigaam-rnnt/` + `/gigaam-audio/example.wav`.

### Startup concurrency probe (2026-08-29)

On the same Chrome headless/NVIDIA Blackwell fixture, three fresh all-WebGPU
loads measured a serial median of `8,821.245 ms` versus an opt-in parallel
median of `7,556.690 ms` (`14.3353%` lower); all six runs preserved exact
transcript parity. The transcribe medians were `4,360.275 ms` and `4,226.870
ms`, respectively, so this is a startup-only signal rather than an
end-to-end throughput claim. Keep the flag off for mixed/WASM compositions
and repeat on another adapter before any default change. Structured evidence:
`docs/reports/gigaam-rnnt-session-init-concurrency-2026-08-29.json`.

Limitation: **Russian-only** punctuation model. No English JFK claim. No preset.

### Commands

```powershell
$env:GIGAAM_RNNT_ONNX_SMOKE='1'
npx vitest run tests/gigaam-rnnt-onnx-backends.test.ts
# Chrome WebGPU (Vite :8765):
# cd N:\github\asrjs\webgpu-agent-test
# node scripts/run-gigaam-rnnt-webgpu.mjs
```

---

## Latency HUD (library + streaming-demo)

`N:\github\asrjs\streaming-demo` now always passes a `transcribe` wrapper into
`createBrowserRealtimeMicrophoneController`, so the inner
`RealtimeTranscriptionController` exists with `latency: true`.

- Demo `App.jsx` queues `queueMicrophoneTranscription(request.pcm, …)` and returns a Promise.
- Clip labels use `request.segmentReason ?? request.reason`.
- Library `publishCapturedAudio` still fires `onUtterance` for PCM/UI, then `transcribeUtterance`, then `emit()` so React reads `getState().latency`.
- HUD: Capture panel **Advanced** → `StreamingDebugHud` maps
  `lastFirstPartialLatencyMs`, `lastEndOfUtteranceLatencyMs`,
  `p50ProcessLatencyMs`, `p95EmitLagMs`.
- Unit tests: `tests/realtime-latency.test.ts`, `tests/browser-controller.test.ts`,
  `tests/browser-realtime.test.ts`.

Idle HUD showing `--` is expected until a **completed** microphone utterance.
That requires real `getUserMedia` audio. Do not claim a human speech pass from
an automation environment that cannot grant mic permission.

2026-08-27 verification (this slice): demo on `http://localhost:3000/` loaded.
Capture **Advanced** opened `StreamingDebugHud` with snapshot `diagnostics · idle · 16000 Hz`
and latency fields `first partial --` / `eou --` / `p50 process --` / `p95 emit --`.
**Start Mic** returned `Microphone error: Permission denied`. Cursor browser MCP
(`cursor-ide-browser`) could create a tab but immediately lost it (`No browser tab
available` / `Browser view not found`); HUD check used system Chrome via CDP instead.
This is **mic-blocked**, not a human speech pass. No demo wiring fix was required.

## Worker cancellation signal compatibility (2026-08-28)

`createBrowserTranscriptionWorkerClient()` now observes all signal shapes
accepted by its public contract: native `AbortSignal` events, structurally
compatible cross-realm abort events, and minimal `{ aborted }` signals through
a 25 ms polling fallback. Cancellation still sends `CANCEL_TRANSCRIBE`,
rejects only the caller's request, keeps the worker/model alive, and allows the
next transcription to reuse the loaded model.

Regression coverage includes the browser transport, worker-thread active and
queued cancellation, and the realtime controller reset path. The minimal
signal case was previously able to leave an in-flight worker request pending;
the test now proves that it is canceled without worker teardown.

## Realtime abort-like signal compatibility (2026-08-28)

The abort observer is now shared by the browser transcription worker client,
`StreamingSpeechDetector.start()`, TEN-VAD init, and FireRed VAD init. Native
and cross-realm abort events remain event-driven; minimal `{ aborted }` objects
are observed with the same 25 ms polling fallback. VAD init timeout wrappers
also remove their observer on every settle path, including timeout, abort,
success, and worker failure.

Regression coverage proves that a minimal signal changed during an in-flight
TEN-VAD init, FireRed VAD init, or streaming detector start reaches the same
idle/teardown contract as a native `AbortSignal`. Core validation after this
slice: 957 passed, 15 skipped; typecheck, build, and lint (0 errors) pass.

## X-ASR public streaming artifact validation (2026-08-28)

The artifact-gated `tests/x-asr-onnx-backends.test.ts` suite now constructs the
public X-ASR model family with the official local Zipformer2 artifacts and
exercises `model.createStreamingTranscriber()` through `pushAudio()` and
`finalize()`, followed by transcriber and model disposal. The 11-second JFK
fixture produced the exact expected transcript. The same suite also retains the
direct executor WASM check and the classified Node WebGPU probe; all three tests
passed on the current workstation. This is public model/session-boundary
evidence for the streaming lifecycle, not a claim of broad X-ASR quality.

Current commit: `f00ce92 test(x-asr): cover public streaming artifact path`.

## Package-level optional batch capability (2026-08-28)

The package now exposes an optional `SpeechSession.transcribeBatch` capability,
with `SpeechBatchSession` as the required-method type for family-specific
consumers. `LoadedSpeechModel` adds `supportsBatch` and a batch method, and
`SpeechPipeline.transcribeBatch` / `transcribeSpeechBatch` reuse the same
cache, lifecycle, response-flavor, and canonical transcript boundaries as
single-item calls.

GigaAM CTC and SenseVoice map each native batch item through their existing
single-item canonical mappers. `canonical`, `native`, and `canonical+native`
therefore have the same meaning for one item and a batch. Batch calls accept
mixed-length short inputs; automatic long-audio windowing remains a separate
per-input runtime operation and is rejected explicitly in a batch call rather
than silently becoming serial inference. Whisper, Qwen, NeMo, and other
families are not advertised as batch-capable until their graph and output
parity contracts are independently proven.

## Qwen long-audio window-merge evidence (2026-08-29)

The official local Qwen3-ASR 0.6B model (`qwen-asr==0.0.6`, CPU, 1024-token
cap) was run against `tests/fixtures/end-of-chapter-4.en.mp3` before comparing
the browser-compatible ONNX path. The source audio is 167.471 seconds,
mono/22050 Hz, SHA-256
`bcb1544bedab93c7ec97734dce50316bbaef5ca59377f75d0f9e1eff4dac784c`; the
paired text label SHA-256 is
`4c47b22678017452ec1f459eeef10adb865ce6ac56f9902c3f8fc5f2a355ac4`.

The generic window merge previously concatenated divergent segment-only
prefixes across overlapping windows. `src/pipeline/long-audio-windowing.ts`
now uses the known temporal overlap to trim a conservative prefix and then
removes any exact overlap that begins after that divergent prefix. The focused
regression tests cover both cases in `tests/pipeline-windowing.test.ts`.

Using the official dynamic encoder and native fp16 decoder graphs in
`N:\models\onnx\qwen3-asr-0.6b-official`, one sequential Node/WASM run with
25-second windows and 5-second overlap produced:

| Metric                                     |   Result |
| ------------------------------------------ | -------: |
| Window count                               |        9 |
| Total time                                 | 627.78 s |
| RTFx                                       |    0.267 |
| Label WER                                  |    4.16% |
| Label CER                                  |    1.53% |
| Implementation vs official-native text WER |    3.73% |

The structured run is
`tools/data/results/qwen/qwen3-asr-0.6b-long-wasm-windowed-end-of-chapter-4-2026-08-29.json`.
The official native reference is
`tools/data/results/qwen/qwen3-asr-0.6b-long-native-reference-end-of-chapter-4-1024-2026-08-29.json`.
The ONNX run used a temporary 16 kHz WAV generated from the immutable MP3 with
FFmpeg (`-ar 16000 -ac 1`); its converted WAV SHA-256 is recorded in the
structured result. This is representative label evidence, not a public Qwen
long-audio quality claim: the family remains short-clip/within-model-limit
experimental support and has no verified encoder-cache streaming contract.

## Deterministic browser microphone acceptance (2026-08-29)

`tools/browser-validation/streaming-demo-mic-smoke.py` now provides a reusable
headless Chromium acceptance probe. Chromium's fake microphone is fed the
speech fixture `tests/fixtures/ItsLifeJim.en.wav` (SHA-256
`720029790d0718aff094b0e1c353d7890bce1c9feba0029f935cd82b3a804e66`) while the
streaming-demo uses its normal `getUserMedia` capture path. The probe selects
the local `parakeet-realtime-eou-120m-v1` artifact, WASM runtime, and
`speech-detect` mode, then verifies model load, worker readiness, a non-empty
transcript, a completed segment, and all four HUD latency fields.

The run is recorded in
`tools/data/results/browser/streaming-demo-mic-smoke-parakeet-realtime-speech-detect-2026-08-29.json`:

| Metric           |  Result |
| ---------------- | ------: |
| Model load       | 4443 ms |
| Segment duration |  2.85 s |
| First partial    | 1899 ms |
| End of utterance | 1086 ms |
| p50 processing   | 1085 ms |
| p95 emit lag     | 1086 ms |

This closes the reproducible fake-device browser acceptance gap for capture →
segmenter → worker/model → HUD. It is not a physical human-microphone pass,
and the fixture has no paired quality label, so neither hardware behavior nor
ASR quality is claimed from this run.

## WebGPU adapter selection resilience (2026-08-29)

`probeWebGpuCapabilities()` now requests the `high-performance` WebGPU adapter
first, which is the appropriate preference for sustained ASR inference. If a
browser or driver returns `null` or rejects that preference, the probe retries
the browser's default adapter selection and preserves the fallback reason in
`capabilities.notes`. This avoids classifying WebGPU as unavailable solely
because the preferred adapter selection is unsupported.

The backend unit suite covers both fallback forms (`null` and rejected option),
and a system-Chrome browser module smoke imported the built source against a
mocked `navigator.gpu`: high-performance selection returned `null`, default
selection succeeded, and the probe reported WebGPU/FP16 available. This is
adapter-selection coverage, not a hardware performance claim.

## Current artifact-gated WASM rerun (2026-08-29)

The official/local artifacts were rerun against the current source after the
adapter probe change. Functional WASM assertions remained green:

| Family                  | Current functional evidence                                                      |
| ----------------------- | -------------------------------------------------------------------------------- |
| GigaAM multilingual CTC | fp32 JFK exact, fp16 JFK exact, and mixed-length batch output exact/non-empty    |
| GigaAM v3 E2E RNN-T     | Official Russian `example.wav` exact                                             |
| SenseVoiceSmall         | JFK exact with `en` metadata and mixed-length batch output exact/non-empty       |
| X-ASR zh-en 160 ms      | JFK exact through both direct WASM and the public stateful streaming transcriber |

The accompanying Node/WebGPU cases still classify this workstation as
`WEBGPU_NO_ADAPTER`, which is an environment result rather than an inference
failure. No Qwen long-audio rerun was needed in this slice; its existing
official-native comparison and label-backed windowed result remain the
authoritative long-audio evidence.

## Node WebGPU parity via onnxruntime-node (2026-08-29)

The earlier `WEBGPU_NO_ADAPTER` classification was an artifact of the shared
executors loading `onnxruntime-web` inside Node: that build resolves WebGPU
through `navigator.gpu`, which Node does not provide, while the native
`onnxruntime-node` package ships its own wgpu adapter. `initOrt` and
`initQwenOrt` now prefer `onnxruntime-node` for WebGPU backends in Node-like
runtimes and fall back to `onnxruntime-web` when the native package is
missing, so browser behavior and CI classifications are unchanged.

Node sessions pass a plain `webgpu` execution provider string and keep all
outputs on the CPU because the native build does not implement gpu-buffer
output locations; KV caches therefore stay CPU-resident in Node runs while
GPU compute executes on the adapter. Colocated `.onnx.data` files are left to
the native loader, because the Node binding only accepts byte buffers for
explicit `externalData` options.

Fresh Node WebGPU evidence on the same official artifacts, exact transcript
match on real GPU:

| Family                  | Node WebGPU evidence                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| GigaAM multilingual CTC | `tools/data/results/gigaam/multilingual-ctc-jfk-short-webgpu.json` (fp32)                         |
| GigaAM v3 E2E RNN-T     | `tools/data/results/gigaam/v3-e2e-rnnt-example-webgpu.json` (fp32, exact Russian)                 |
| SenseVoiceSmall         | `tools/data/results/sensevoice/sensevoice-small-jfk-short-webgpu.json`                            |
| X-ASR zh-en 160 ms      | `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu.json`                                |
| Qwen3-ASR 0.6B          | `tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-webgpu.json` (official fp32, RTFx 1.41, CPU KV) |

Classified limitation: the pinned `onnxruntime-node` nightly requires
`Float16Array` backing for float16 tensors, but its native buffer extraction
cannot read that type, so fp16 graphs fail on every EP with a
"not enough space: expected N, got 0" binding error. The Node WebGPU Qwen
probe therefore uses the official fp32 decoder graphs; fp16 remains
browser-only until the ORT-node binding supports Float16Array buffers.

## Browser WebGPU optimization pass (2026-08-29)

Browser performance is measured through the existing
`N:\github\asrjs\webgpu-agent-test` system-Chrome harness, not the Node WebGPU
binding. The harness now bundles and serves the exact ORT Web version installed
by this library (`1.27.0-dev.20260506-673c3320fc`) instead of silently
overriding it with its stale ORT 1.26 copy. Result payloads record the engine and
version. A Vite regression from literal `import('onnxruntime-node')` calls was
also fixed: Node package loading now crosses the browser-safe `node-compat`
bridge, so the harness build no longer tries to bundle native `.node` binaries.

The workstation also had SportsQuant bound to `127.0.0.1:8765` while Vite held
the wildcard listener. Harness runners now accept `ASRJS_WEBGPU_HOST`; the runs
below used `127.0.0.2` and reached Vite deterministically. Production bundle
validation skips copying local multi-gigabyte fixtures (`copyPublicDir: false`)
and passes; dev asset middleware remains the real model-serving path.

### Qwen GPU-resident KV

The Qwen harness had forced `cacheOutputLocation: 'cpu'`, masking the library's
existing WebGPU default (`gpu-buffer`). Both variants were run three times on
one loaded executor, with exact oracle text on every run:

| Qwen3-ASR 0.6B        |       CPU KV |       GPU KV (default) |
| --------------------- | -----------: | ---------------------: |
| Median inference      | 7,115.205 ms |           4,856.355 ms |
| Median RTFx           |       1.546x |                 2.265x |
| Median wall reduction |            - | 31.75% (1.47x speedup) |

Evidence:

- `tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-webgpu-cpu-kv-chrome.json`
- `tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-webgpu-chrome.json`

### X-ASR GPU-resident encoder state

The X-ASR 160 ms encoder returned 116 cache tensors to CPU after every tiny
streaming step, then uploaded them again on the next step. The family-specific
default graph now names those state outputs, and the shared ORT session bridge
supports a per-output location map. Only `encoder_out` is downloaded; cache
outputs remain as GPU buffers and are passed directly into the next encoder
step. Exact transcript parity is preserved:

| X-ASR zh-en 160 ms |     CPU state |    GPU state (default) |
| ------------------ | ------------: | ---------------------: |
| Inference          | 32,095.140 ms |           9,263.655 ms |
| RTFx               |        0.343x |                 1.187x |
| Wall reduction     |             - | 71.14% (3.46x speedup) |

Evidence:

- `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-cpu-state-chrome.json`
- `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-chrome.json`

The fresh Node/WASM evidence is still faster for this small, high-frequency
graph (6.84 s), so the result does not justify a universal WebGPU preference.
It closes most of the WebGPU state-transfer penalty and leaves backend choice
artifact- and workload-specific.

The artifact matrix also exposed and closed a Node-hosted ORT Web regression:
colocated Qwen decoder external data was incorrectly omitted for the WASM
engine as though native ORT would resolve it. WASM now mounts the external data
bytes explicitly, with focused regression tests in both the Qwen and shared
ORT bridges. The real fp16 Qwen WASM decoder (1.50 GB external data) and the
paired Node WebGPU leg both pass with exact text after the fix.

GigaAM CTC, SenseVoiceSmall, and GigaAM RNN-T were also rerun through Chrome
after ORT alignment and remained exact. Their single-run timings are retained
as compatibility evidence, not promoted as before/after performance claims.
ORT Web 1.29.0 was initially identified as the current stable candidate while
npm on this host returned `ETARGET` and the registry tarball remained reachable.
The follow-up resolved the lock metadata, pinned `onnxruntime-web` and
`onnxruntime-common` to 1.29.0, and validated offline install, build, full
tests, and the real-artifact browser matrix. See the lockfile follow-up in
`docs/GOAL_PROMPT.md` and commit `2ed88b9`.

## X-ASR incremental frontend optimization (2026-08-29)

The X-ASR streaming executor previously rebuilt the complete accumulated audio
feature matrix on every `pushStream()` call. The family-specific fbank frontend
now processes only newly sample-backed frames and retains a bounded 400-sample
raw-audio tail. With `snip_edges=false`, right-edge reflected frames are held
until the next chunk or finalization; this is required because their values
change as future samples arrive.

The uneven-chunk parity test is exact (`maxAbs=0`). The executor also keeps the
exact logical cumulative audio view while appending into amortized backing
storage, so repeated pushes no longer allocate and copy the complete audio
buffer. The Node CPU microbenchmark uses deterministic synthetic audio, 200 ms
chunks, three timed runs after one warm-up, and compares the former
full-buffer-per-chunk loop with the incremental loop:

| Audio | Chunks | Full recompute median | Incremental median | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2 s | 10 | 22.8828 ms | 5.0927 ms | 4.4933x |
| 10 s | 50 | 543.0221 ms | 22.9987 ms | 23.6110x |

This is frontend/storage CPU evidence, not an end-to-end RTFx claim. The
frontend-only controls measured 4.5512x and 17.5392x speedups at 2 s and 10 s;
the combined candidate uses amortized capacity growth while preserving the
logical `Float32Array` view. Retained capacity can exceed logical length until
stream disposal. Reproduce with
`npm run benchmark:x-asr-frontend -- --runs=3 --durations=2,10 --json`.
Evidence: `docs/reports/x-asr-incremental-frontend-benchmark-2026-08-29.json`.

The opt-in streaming browser checkpoint also passes on the local real artifact:
Chrome headless/WebGPU, ORT Web 1.29.0, NVIDIA Blackwell, 55 x 200 ms chunks,
exact oracle text, 8,981.14 ms (`1.2248x` RTFx). This is a parity and end-to-end
behavior checkpoint rather than a browser before/after claim. Evidence:
`docs/reports/x-asr-webgpu-streaming-parity-2026-08-29.json` and
`tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-stream-chrome.json`.

## Remaining gaps

- Dynamic encoder is now the default official-graph load (library helper + Chrome/Qwen harness). Static T=1100 remains opt-in via `encoder=static-t1100` / `QWEN_OFFICIAL_ENCODER=static-t1100`.
- Qwen long-audio is now covered by one official-native comparison and one
  label-backed window-merge run; broader language/domain and repeated-run
  coverage remains open, so it is still not a promotion gate.
- Node WebGPU now passes through `onnxruntime-node` for all five families
  (2026-08-29); Node KV caches stay on the CPU and fp16 graphs remain blocked
  by the ORT-node Float16Array binding gap. This is non-urgent and is not a
  browser promotion gate; real browser WebGPU remains the performance path.
- ORT Web 1.29 stable upgrade is complete and validated. Keep the separate
  `onnxruntime-node` nightly dependency isolated; its Float16Array binding gap
  is non-urgent and not a browser promotion gate.
- All five families stay experimental; no presets. Discover via `listExperimentalSpeechFamilies()`, not `listSpeechModels()`.
- GigaAM RNN-T is Russian-only (`example.wav`); do not cite it as a JFK / English result.
- Deterministic fake-device browser acceptance for the streaming-demo latency HUD is now recorded; a physical human-microphone pass remains a manual device check. `--` at idle is not a speech-path pass.
- Historical note: this handoff originally recorded experimental families on the root for WebGPU discovery. Current main keeps them on intentional `models/<family>` subpaths; they remain experimental and are not presets.
- X-ASR now owns its native transcript/options contracts and canonical mapper under `src/models/x-asr`; it no longer reuses the LASR-CTC family contract.
- Realtime transcription requests now carry a controller-owned abort signal; `reset()` aborts stale in-flight callbacks before clearing state, so browser worker/model callbacks can cancel cooperatively and reuse the loaded model.
- Experimental family executors now call ORT `session.release()` on dispose. Whisper encoder mel feeds, decoder-step owned feeds/logits/replaced KV (CPU+GPU), and split-graph callback present-KV Ort wrappers are `dispose()`d after copy; next-step encoder KV is retained until replaced. CTC/transducer `session.run()` output logits (GigaAM CTC/RNN-T, SenseVoice, X-ASR, LASR, Wav2Vec2, NeMo TDT/RNN-T/AED) and Qwen prefill/step decoder logits are copied then disposed; Qwen next-step KV is retained until replaced. Browser capture worklet URLs, decode AudioContext `close()`, and TEN/FireRed VAD `worker.terminate()` on adapter dispose were already present. `UrlAssetHandle` / `BlobAssetHandle` now refuse post-dispose locators and revoke blob URLs on dispose even when `getLocator('url')` is concurrent or still in-flight. `getModelFile({ preferBlobUrl: true })` now requires `onResolvedHandle` and always disposes the handle when ownership is not transferred. Whisper, Qwen, GigaAM RNN-T joint/decoder, NeMo AED decoder, NeMo RNN-T joint/decoder, NeMo TDT duration/step, and X-ASR streaming step loops honor `options.signal` between steps (`PipelineAbortedError`). X-ASR abort does not commit the in-flight chunk or dispose caller encoder-state tensors; the leftover stream can be retried (`reset()` optional). Experimental families now expose structured `languages` / `audioContract` / `limitations` on `listExperimentalSpeechFamilies()`, and missing local ONNX throws root-exported `ExperimentalArtifactMissingError` (`code === 'experimental-artifact-missing'`, `isExperimentalArtifactMissingError()`). `loadSpeechModel({ signal })` aborts with `PipelineAbortedError('load')`. Remaining product gaps: physical human-microphone check and Qwen long-audio; ORT-node Float16Array binding remains optional backlog work.
