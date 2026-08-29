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

## Current-checkout parity refresh (2026-08-28)

The official-artifact smoke was rerun against the current checkout at commit
`0ba481b` using `audio-encoder-dynamic.onnx`, the fp16 prefill/step graphs, and
the local external-data files. The 10.5-second JFK fixture matched the native
Qwen oracle exactly (`text_match=true`, 30 emitted tokens), with WASM metrics of
29.522 s total, RTFx 0.3567, 3.949 s encoder time, 14.004 s decoder time, and
approximately 4,337 MiB resident memory. This is an artifact-local parity
refresh, not a representative benchmark or a public preset gate.

The same run recorded Node WebGPU as `WEBGPU_NO_ADAPTER`; this host-level
adapter limitation remains separate from the successful Chrome WebGPU evidence.

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

## Browser WebGPU bundle boundary refresh (2026-08-29)

A fresh Chrome headless rerun initially produced a reproducible corrupted
13-token transcript on ORT Web 1.29.0 even though the Qwen sessions created
successfully. Artifact hashes and the library source matched the earlier exact
run. The failing difference was in the sibling browser harness: it aliased
both `onnxruntime-web` and `onnxruntime-web/webgpu` to
`ort.all.bundle.min.mjs`. Mapping the WebGPU subpath to the all-backend bundle
is not a safe provider substitution for autoregressive graphs.

The harness now keeps the imports separate: plain `onnxruntime-web` resolves
to `ort.all.bundle.min.mjs`, while `onnxruntime-web/webgpu` resolves to
`ort.webgpu.min.mjs`. On the same official dynamic encoder and fp32 explicit-KV
decoder artifacts, NVIDIA Blackwell, and ORT Web 1.29.0, the corrected path
restored exact 30-token parity. Three same-session GPU-KV runs measured
`2686.75`, `1854.60`, and `1792.43` ms (median `1854.60` ms, `5.93x` RTFx);
the CPU-KV control measured `4885.94`, `3880.65`, and `3802.04` ms (median
`3880.65` ms, `2.83x` RTFx). Every run was exact. The GPU-KV median was
`2.09x` faster than CPU-KV on this session.

This is a harness/runtime-entry correction, not a decoder graph or model
algorithm change. Keep the alias separation as a browser acceptance invariant
and repeat it on another browser/adapter before changing a public preset.
Machine-readable evidence is in
`tools/data/results/qwen/qwen3-asr-webgpu-bundle-boundary-2026-08-29.json`.

## Decoder phase profile and allocation hypothesis (2026-08-29)

The Qwen executor now exposes model-specific decoder phase buckets for both
the official stacked graphs and the legacy per-layer graph:
`decoderInitInputMs`, `decoderInitRunMs`, `decoderInitOutputMs`,
`decoderStepFeedBuildMs`, `decoderStepRunMs`, and `decoderStepOutputMs`.
This makes the hot loop measurable without changing the transcript contract or
cache ownership. The instrumentation was validated with the focused Qwen test
set (20/20) and a real Chrome/WebGPU run.

On the exact official dynamic encoder plus stacked prefill/step artifacts,
Chrome headless, NVIDIA Blackwell, and ORT Web 1.29.0, three same-session
GPU-KV runs were exact (30 tokens) with a `1753.825 ms` median transcription
time (`6.2722x` RTFx). The median decoder-step profile was:

| Phase | Median | Share of decoder steps |
| --- | ---: | ---: |
| Step feed construction | `0.235 ms` | `0.0152%` |
| ORT `session.run()` | `1520.570 ms` | `98.43%` |
| Logit/state output handling | `24.240 ms` | `1.57%` |
| All decoder steps | `1544.880 ms` | `100%` |

The CPU-KV control was also exact and measured `3885.960 ms` median (`2.8308x`
RTFx), so GPU-resident KV remains the material placement win (`2.2157x` total
speedup; `2.3604x` on the step loop). Reusing the one-element input tensors or
typed-array wrappers was therefore rejected: the measured feed-build share is
too small to justify additional mutable-tensor ownership risk. Future Qwen
optimization should target the decoder-step graph and WebGPU EP execution
(fusion, graph capture, dispatch, and kernel behavior), with exact-token and
disposal controls around every experiment.

Machine-readable evidence is in
`docs/reports/qwen3-asr-webgpu-decoder-profile-2026-08-29.json` and the tracked
Chrome controls in `tools/data/results/qwen/`.

## Decoder graph-capture probe (2026-08-29)

Qwen direct and Hugging Face artifact sources now accept the diagnostic-only
`decoderGraphCapture` flag and optional `decoderFreeDimensionOverrides`. The
session helper mirrors Whisper's narrow behavior: request capture only on
WebGPU, retry without it only when ORT reports a graph-capture/partitioning
failure, and surface a recoverable warning. The production default remains
unchanged.

The real Chrome probe requested capture for both the official prefill and
decoder-step sessions. ORT Web 1.29.0 rejected both because not all graph nodes
were partitioned to `WebGpuExecutionProvider`; the fallback then completed with
exact 30-token parity. The capture request took `62519.905 ms` to load versus
`35737.84 ms` for the regular control, and the one captured run was only
`4.3656x` RTFx. This is a compatibility boundary, not a performance win.

The decoder-step artifact also has dynamic `past_len`/`present_len` dimensions,
so static-shape graph capture would require a new export and a memory-traffic
comparison. Do not force static KV shapes or enable capture by default until a
future export or ORT EP partitions the complete graph and exact-token,
repeated-run, and disposal checks pass.

Machine-readable evidence:
`docs/reports/qwen3-asr-webgpu-graph-capture-2026-08-29.json`.

## Decoder ArgMax graph-surgery probe (2026-08-29)

The measured decoder profile showed that the library only needs one greedy
token id, while the official prefill/step graphs expose a complete
151,936-wide `logits` output. The reusable
`tools/model-debugging/reference/qwen3-asr-0.6b/append_argmax_output.py`
tool now appends an ONNX `ArgMax(axis=-1, keepdims=1)` output named
`next_token_id`. It can retain logits for an A/B control or remove the logits
graph output for the scalar-fetch candidate; both variants reuse the original
external-data shards without overwriting them.

The Qwen executor accepts the optional scalar output for both stacked and
legacy decoder loops. It reads an INT64/INT32 scalar directly and validates it
against the configured vocabulary. Unmodified graphs continue through the
owned float32 logits copy and JavaScript argmax path. Preferred-output maps
explicitly keep `next_token_id` on CPU while KV tensors remain at the selected
cache location.

The float32 candidate was checked with native `onnxruntime-node` CPU session
creation and then run through the same Chrome headless/NVIDIA Blackwell/
ORT Web 1.29.0 WebGPU harness as the regular graph. Both prefill and step
graphs loaded with outputs `[present_keys, present_values, next_token_id]` and
the complete 30-token JFK transcript was exact. Five-run same-session controls
(runs 2–5 as warmed samples) measured:

| Metric                         | Official logits output | ArgMax-only candidate |
| ------------------------------ | ---------------------: | --------------------: |
| Load time                      |          37,622.665 ms |         38,678.630 ms |
| Median transcription           |           1,623.058 ms |          2,619.933 ms |
| Median RTFx                    |                6.7778× |               4.1992× |
| Median decoder `session.run()` |           1,420.995 ms |          2,374.170 ms |
| Median output handling         |              22.225 ms |              0.655 ms |

The scalar readback itself is 97.05% cheaper, but the added reduction kernel /
graph plan increases decoder execution 67.08% and total transcription 61.42%
on this artifact, so the candidate is classified
`PERFORMANCE_NOT_VIABLE` and is not a production default. This is a useful
negative result: minimizing a transfer does not guarantee a WebGPU win when
the provider must perform a full-vocabulary reduction. Revisit only after a
provider-level ArgMax fusion/reduction improvement or a model graph that can
produce a token without an expensive GPU reduction.

Machine-readable evidence:
`docs/reports/qwen3-asr-webgpu-argmax-surgery-2026-08-29.json` and the paired
browser captures in `tools/data/results/qwen/`.
