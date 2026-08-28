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

The code and mocked graph boundary pass locally. A local official snapshot and
native reference JSON are now available for one fixed 26.45-second fixture;
the result below is still an artifact-local acceptance run, not a hosted or
representative benchmark. The remaining verification work is:

1. run the offline `capture_qwen_reference.py` tool on more fixed audio
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
offline native capture is recorded at
`tools/data/results/qwen/qwen3-asr-0.6b-long-native-reference-2026-08-28.json`
with audio SHA-256
`58ce74b97dfb2c459966baf899a98e217d14130f23dc431b8b86aba121da4335`.

On that fixture, native `qwen-asr` 0.0.6 CPU/float32 completed in 15.948 s.
The official dynamic-encoder/FP16-decoder/WASM library path, with default
windowing disabled by the 30-second model limit (one 26.45-second request),
completed in 63.183 s at RTFx 0.419 and matched the native oracle exactly
(WER 0%, CER 0%, normalized exact match). This is an artifact-local
acceptance result, not a hosted or representative benchmark.

The same path with explicitly forced 10-second windows and 2-second overlap
completed in 105.899 s at RTFx 0.250 and scored WER 3.51% / CER 2.85% against
the native oracle. This documents a real limitation of segment-only overlap
composition: Qwen has no word timestamps in this graph, so forced windows can
change boundary text even when the direct within-limit route is exact.

2026-08-28 long-audio label comparison: the benchmark now auto-loads an
adjacent fixture JSON (or accepts `--reference`) and reports Unicode-safe WER /
CER as `fixture-sidecar-dataset-label`, never as an official Qwen oracle. On
`00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav`
with the official dynamic encoder, native-fp16 decoder, WASM, no warmup, one
run, and forced 10-second windows with 2-second overlap:

| Measurement                      | Before segment-overlap merge | After segment-overlap merge |
| -------------------------------- | ---------------------------: | --------------------------: |
| WER against sidecar `normalized` |                       28.36% |                      25.37% |
| CER against sidecar `normalized` |                       17.19% |                      13.75% |
| elapsed time                     |                   105.9999 s |                  106.2819 s |
| composed windows / decoder steps |                      4 / 105 |                     4 / 105 |

The after output removes duplicated overlap phrases such as `Flow of the ITI`
and `Terminates at the OSG`; it still ends with an extra `central` token. The
native-oracle run shows that this is introduced by the forced window route,
not by the within-limit direct Qwen graph. Official oracle coverage beyond the
30-second model limit, WebGPU validation, and human-microphone validation
remain open. The sidecar comparison remains a local dataset/TTS label and is
not a model-quality claim.
