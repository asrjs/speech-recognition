# Qwen3-ASR-0.6B browser backend handoff

Date: 2026-08-26
Workspace: `N:\github\asrjs\speech-recognition`

## Decision

Qwen3-ASR-0.6B remains worth one bounded integration pass, but it is not a
new ONNX-conversion project. Parakeet TDT v3 is the working browser reference
for this package. Qwen is valuable only if its multilingual speech-LLM
quality, Turkish behavior, or model-family coverage justifies the extra
autoregressive decoder cost.

The current public ecosystem has two useful but different artifacts:

- The official model and reference package are
  [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and
  `qwen-asr`.
- [`goryodog/tokihisu-qwen3-asr-0.6b-webgpu`](https://huggingface.co/goryodog/tokihisu-qwen3-asr-0.6b-webgpu)
  already provides a browser-oriented ONNX encoder plus merged decoder with
  external data and a GPU-KV contract.
- [`andrewleech/qwen3-asr-onnx`](https://github.com/andrewleech/qwen3-asr-onnx)
  provides a separate export/validation path with decoder-init and
  decoder-step graphs and useful CPU/WebGPU measurements.

Neither artifact is an `@asrjs/speech-recognition` backend: neither supplies
this package's canonical transcript mapping, asset lifecycle, progress hooks,
long-audio boundary, browser regression harness, or reference-lineage report.
The implementation in this branch therefore connects an explicit artifact to
the library instead of re-exporting another project's UI or architecture.

## Implemented boundary

The new `src/models/qwen-asr` family contains:

- the exact 128-bin, 16 kHz, 400-point FFT / 160-hop frontend contract,
  minimum-input padding, 100-frame feature padding, 800-frame graph padding,
  and `input_features_mask`;
- a Qwen GPT/ByteLevel BPE tokenizer that preserves chat/audio special tokens;
- direct and Hugging Face artifact sources, external-data mapping, progress
  events, and WebGPU/WASM ORT session setup;
- the conversion's audio encoder output-mask selection;
- prefill plus one-token decoder execution with 28-layer explicit KV cache,
  CPU logits, optional GPU-buffer cache placement, EOS handling, metrics,
  cleanup, and canonical transcript mapping;
- a strict no-source boundary: no scaffold transcript is returned when a real
  artifact source is absent;
- deterministic frontend, tokenizer, model-family, and mocked prefill/KV-step
  tests in `tests/qwen3-asr.test.ts`.

The graph is currently treated as batch-1. Batch inference belongs in a later
graph contract; exposing a `batchSize` option now would imply support that the
published browser graph does not have. Word timestamps are also deliberately
not inferred. Qwen's separate ForcedAligner must be integrated as a separate
artifact and verification boundary.

## Artifact contract recorded from the WebGPU conversion

The reference conversion declares:

| Component      | Contract                                                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| Encoder input  | `input_features` float16 `[1,128,audio_frames]`; `input_features_mask` int32 `[1,audio_frames]`                   |
| Encoder output | `audio_embeddings` float16 `[1,audio_tokens,1024]`; `audio_token_mask` bool                                       |
| Decoder input  | int32 IDs; float16 audio embeddings/mask/attention; int32 positions; 28 × key/value past tensors `[1,8,past,128]` |
| Decoder output | logits plus 28 × `present.*` key/value tensors                                                                    |
| Audio graph    | frames padded to a multiple of 800; 104 audio placeholders per full 800-frame window                              |
| Cache          | seed length 1, zero seed values, seed attention mask `-65504`; GPU-buffer cache when decoder WebGPU is available  |
| Stopping       | EOS IDs `151643` and `151645`; pad ID `151645`                                                                    |

The external data files are large (roughly 373 MiB encoder and 1.5 GiB
decoder in the public artifact). They are never downloaded or committed by
this work. A caller must explicitly pass the source and accept the artifact
license/provenance.

## Verification gate

The code and mocked graph boundary pass locally, but the real model is not
yet marked end-to-end verified because no approved local Qwen snapshot or
reference JSON exists in this workspace. The next run, after an approved
artifact is available, is:

1. run the offline `capture_qwen_reference.py` tool on several fixed audio
   samples, recording audio hashes, language, text, and optional alignments;
2. run the same sample through native ORT, WASM, and WebGPU with the exact
   conversion manifest;
3. compare frontend stats, encoder mask/token count, first decoder logits,
   first five token IDs, EOS, final text, and repeated-run cleanup;
4. record load time, peak memory, encoder/decode split, KV location, RTFx, and
   browser/adapter details before deciding whether Qwen deserves a preset.

For a local ONNX/WASM baseline once the artifact is approved:

```powershell
npm run build
node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs `
  --model-dir N:\models\onnx\qwen3-asr-0.6b-official `
  --audio tests\fixtures\jfk2.en.wav `
  --backend wasm --warmup 1 --runs 3
```

The harness requires all graph, external-data, and tokenizer files locally;
it has no implicit model download path. It accepts both the current official
dynamic/static encoder plus prefill/step graph layout and the older merged
decoder layout. To measure the library long-audio route explicitly, use a
forced model-safe window, for example:

```powershell
node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs `
  --model-dir N:\models\onnx\qwen3-asr-0.6b-official `
  --audio tests\fixtures\00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav `
  --backend wasm --warmup 0 --runs 1 --window-seconds 10 --overlap-seconds 2
```

That route is a runtime/windowing compatibility check unless the audio has a
separately captured oracle transcript; its output must not be presented as a
quality score by itself.

If this gate shows no quality or compatibility advantage over Parakeet v3,
close Qwen as a documented candidate rather than expanding the public API.

2026-08-28 continuation: the public `LoadedSpeechModel.transcribeMonoPcm()`
path now uses the same model-aware windowing planner as `transcribe()`. The
official dynamic/FP16 benchmark passed on the 26.45-second fixture with four
forced windows (109.2 seconds total, RTFx 0.242). The short JFK oracle also
remained exact (11 seconds, 32.97 seconds total, RTFx 0.334). These are local
measurements, not hosted CI claims; the long medical fixture has no gold text.
