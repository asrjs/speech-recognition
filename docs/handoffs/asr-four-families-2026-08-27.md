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

| Family | Official source | Oracle |
| --- | --- | --- |
| GigaAM CTC | GigaAM `multilingual_ctc` + `to_onnx` | official PyTorch / native ORT |
| SenseVoiceSmall | FunAudioLLM `model.export` | FunASR, not OpenVoiceOS ONNX |
| X-ASR zh-en 160ms | sherpa-onnx Zipformer2 streaming | sherpa; JS fbank = knf `snip_edges=false`, `high_freq=-400` |
| Qwen3-ASR 0.6B | `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`, `qwen-asr==0.0.6` | official CPU; third-party ONNX is not the oracle |

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

| Family | Chrome WebGPU | WASM |
| --- | --- | --- |
| GigaAM CTC | exact, official fp16 | exact (fp32 and official fp16) |
| SenseVoiceSmall | exact | exact |
| X-ASR zh-en 160ms | exact | exact (same as sherpa, no extra comma) |
| Qwen3-ASR 0.6B | exact, 34.1s load / 9.8s transcribe | sequential fp16 and fp32 exact; Chrome fp16 JS heap 2082/4192 MB; Chrome fp32 also passed (WASM linear memory is outside `performance.memory`) |

Qwen Node WASM fp16: RSS 4254 MB, 30.7s, RTFx 0.36. Chrome WASM fp16: 46.5s, RTFx 0.24. WebGPU remains the faster browser path.

## Promotion criteria (do not invent presets)

Keep all four **experimental**. A public preset needs:

1. Oracle-matching greedy text on the checked fixture (done for JFK).
2. At least one supported browser path (Chrome WebGPU is that path for all four; Qwen sequential WASM is an extra fallback, not a reason to drop WebGPU).
3. Qwen additionally needs the dynamic encoder in the default load path (done: library `resolveOfficialQwen3AsrDirectArtifacts` and Chrome/Qwen harness default to `audio-encoder-dynamic.onnx`; static T=1100 is opt-in).
4. No third-party ONNX as the Qwen oracle.
5. Streaming/long-audio contracts remain family-specific (X-ASR is true encoder-cache streaming; Qwen is short-clip offline speech-LLM).

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

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `v3_e2e_rnnt_encoder.onnx` | 885,093,282 | `41ef815cebb6cdc7158321ec2a8b4d1ab04d1eb55f1c36ff98b428d39d0866a7` |
| `v3_e2e_rnnt_decoder.onnx` | 4,599,970 | `fa95ac0997e621ebee4156fc9295e57407549e2d5c20ed548626ea1e4f20a09c` |
| `v3_e2e_rnnt_joint.onnx` | 2,712,926 | `2ead0f16a18554b8d557875110da4b1e964441f617cdd95d0b4b45ac999c60a9` |
| `v3_e2e_rnnt_vocab.txt` | 14,379 | `b98730cffb0bb782f505003caff8b4ba03ef6c9baf6b799572af3191f42fe098` |

Graph I/O:

- encoder `audio_signal` / `length` → `encoded` / `encoded_len`
- decoder `x` / `hi` / `ci` → `dec` / `ho` / `co`
- joint `enc` / `dec` → `joint`

Official greedy: blank does **not** update predictor LSTM state. `hi`/`ci` are `[layers, batch, hidden]`.
WebGPU sessions must load **sequentially** (`another WebGPU EP inference session is being created` if `Promise.all`).

### Gates (exact greedy text vs official `example.wav`)

| Gate | Result |
| --- | --- |
| PyTorch `transcribe` | exact |
| Native ORT greedy | exact; encoder max-abs ~3e-6 vs PyTorch |
| Node WASM (`GIGAAM_RNNT_ONNX_SMOKE=1`) | exact |
| Chrome WebGPU (NVIDIA Blackwell) | exact; load ~8.0s, transcribe ~5.3s, RTF 0.47 |
| Node WebGPU | `WEBGPU_NO_ADAPTER` |

Chrome result: `tools/data/results/gigaam/v3-e2e-rnnt-example-webgpu-chrome.json`
Harness: `N:\github\asrjs\webgpu-agent-test` — `gigaam-rnnt.html`, asset route `/gigaam-rnnt/` + `/gigaam-audio/example.wav`.

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

## Remaining gaps

- Dynamic encoder is now the default official-graph load (library helper + Chrome/Qwen harness). Static T=1100 remains opt-in via `encoder=static-t1100` / `QWEN_OFFICIAL_ENCODER=static-t1100`.
- Qwen long-audio / T not from this 11s clip beyond the T=1050 pad-crop identity check.
- Node WebGPU still `WEBGPU_NO_ADAPTER`.
- All five families stay experimental; no presets. Discover via `listExperimentalSpeechFamilies()`, not `listSpeechModels()`.
- GigaAM RNN-T is Russian-only (`example.wav`); do not cite it as a JFK / English result.
- Human microphone pass for the streaming-demo latency HUD is still required. `--` at idle is not a speech-path pass.
- Historical note: this handoff originally recorded experimental families on the root for WebGPU discovery. Current main keeps them on intentional `models/<family>` subpaths; they remain experimental and are not presets.
- X-ASR now owns its native transcript/options contracts and canonical mapper under `src/models/x-asr`; it no longer reuses the LASR-CTC family contract.
- Realtime transcription requests now carry a controller-owned abort signal; `reset()` aborts stale in-flight callbacks before clearing state, so browser worker/model callbacks can cancel cooperatively and reuse the loaded model.
- Experimental family executors now call ORT `session.release()` on dispose. Whisper encoder mel feeds, decoder-step owned feeds/logits/replaced KV (CPU+GPU), and split-graph callback present-KV Ort wrappers are `dispose()`d after copy; next-step encoder KV is retained until replaced. CTC/transducer `session.run()` output logits (GigaAM CTC/RNN-T, SenseVoice, X-ASR, LASR, Wav2Vec2, NeMo TDT/RNN-T/AED) and Qwen prefill/step decoder logits are copied then disposed; Qwen next-step KV is retained until replaced. Browser capture worklet URLs, decode AudioContext `close()`, and TEN/FireRed VAD `worker.terminate()` on adapter dispose were already present. `UrlAssetHandle` / `BlobAssetHandle` now refuse post-dispose locators and revoke blob URLs on dispose even when `getLocator('url')` is concurrent or still in-flight. `getModelFile({ preferBlobUrl: true })` now requires `onResolvedHandle` and always disposes the handle when ownership is not transferred. Whisper, Qwen, GigaAM RNN-T joint/decoder, NeMo AED decoder, NeMo RNN-T joint/decoder, NeMo TDT duration/step, and X-ASR streaming step loops honor `options.signal` between steps (`PipelineAbortedError`). X-ASR abort does not commit the in-flight chunk or dispose caller encoder-state tensors; the leftover stream can be retried (`reset()` optional). Experimental families now expose structured `languages` / `audioContract` / `limitations` on `listExperimentalSpeechFamilies()`, and missing local ONNX throws root-exported `ExperimentalArtifactMissingError` (`code === 'experimental-artifact-missing'`, `isExperimentalArtifactMissingError()`). `loadSpeechModel({ signal })` aborts with `PipelineAbortedError('load')`. Remaining product gaps: human mic HUD, Node WebGPU adapter, Qwen long-audio.
