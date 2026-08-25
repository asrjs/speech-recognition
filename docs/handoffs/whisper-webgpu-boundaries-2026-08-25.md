# Whisper WebGPU Boundary Continuation

Date: 2026-08-25
Workspace: `N:\github\asrjs\speech-recognition`
Browser harness: `N:\github\asrjs\webgpu-agent-test`

## Landed in this continuation

- Split-graph word alignment now follows Whisper/faster-whisper's
  teacher-forced `find_alignment` contract: the alignment graph receives
  `[SOT, language, task, no_timestamps, text..., EOT]`, retaining the
  no-timestamps prediction row as the DTW anchor while generated timestamp
  spans remain available for segment context.
- Encoder hidden states are cast at graph boundaries in either direction
  (`float16` ↔ `float32`) using the graph's declared input metadata. This is
  required when the browser uses the fp16 encoder with a corrected fp32
  `decoder_align` export.
- The ONNX alignment exporter now supplies the decoder's causal self-attention
  mask and uses the installed Transformers 4.x attention-call contract.
- Leading DTW pauses are preserved by duration postprocessing instead of
  being clipped back to zero by the short-word guard.
- The WhisperX-compatible Node runner is portable on Windows: local dynamic
  imports are converted to `file:` URLs and FFmpeg stderr uses `NUL` instead
  of `/dev/null`.
- Runner decoder KV entries retain their own `{ data, dims, type }`. Stable
  beam hypotheses therefore keep the correct sequence length after another
  hypothesis advances; a shared global shape map is only a compatibility
  fallback.
- Batched beam remains opt-in. If a backend rejects a batch-shaped decoder
  step, the active hypotheses are retried through the scalar path and the
  optimization is disabled for the rest of that decode. Wrong result counts
  still fail loudly.
- Stable beam expansion now shares immutable parent KV-cache objects. The
  decoder bridge owns cloning/repacking when it builds the next input, so the
  core no longer duplicates every layer once per sibling hypothesis.
- Split-graph manifests now carry an `alignment_export` marker. A manifest
  without `causal_self_attention: true` is treated as a legacy alignment
  artifact: the runtime emits recoverable warnings and uses generated
  timestamp interpolation instead of claiming verified word alignment.
- The exporter supports the installed Transformers 4.41 legacy cache API,
  accepts a local model-snapshot path for tokenizer/config files, and makes
  `--external-data always` externalize every initializer. This keeps large
  encoder graphs browser-loadable and makes manifest external-data metadata
  reflect the files that actually exist.
- The GPU-KV policy guard runs before mel preprocessing and encoder inference.
  GPU-KV is still greedy argmax only; beam, `best_of`, and temperature remain
  rejected until cache cloning/reordering is proven correct.
- Whisper mel preprocessing now defaults to the exact N_FFT=400 contract. The
  cached Bluestein implementation is still fast enough for the runtime; the
  512-point radix-2 path remains available only as `fastFft: true` for explicit
  experiments because its frequency-bin grid is not model-parity compatible.
- Encoder `input_features` are cast from float32 to float16 when a corrected
  fp16 export declares that input type. This keeps the preprocessor dtype-neutral
  while making the graph boundary explicit.
- The merged-decoder forced-alignment path now shares the same reference prompt
  builder and task selection as split-graph alignment. It derives the prompt
  rows, filters cache feeds from the decoder's declared inputs, casts encoder
  states at the merged decoder boundary, reads GPU attention/logit outputs
  safely, and aligns the no-timestamps anchor plus text rows against the
  attention graph's frame axis.
- Merged-decoder alignment also crops padded frames to the actual audio duration;
  it no longer halves the encoder hidden-state length as a proxy for attention
  frames. Focused regression tests cover the prompt, causal logit rows, cache
  inputs, text-row extraction, and duration crop.

## Validation

Current focused coverage: 43 alignment, timestamp, and forced-prompt tests
passed. Typecheck and build passed.

Real local runner artifacts:

- English `jfk-10s.ogg`, custom `fp16_iofp32` splitgraph model, beam 2:
  Windows OGG conversion, Node inference, and coherent transcript completed.
- Turkish `tr-tdk-18s.wav`, the same model, `language=auto`, beam 2:
  language detection selected `tr` and produced coherent Turkish text.
- English beam 2 with decoder-align word timestamps completed with 16 words.

Independent headless Chrome/WebGPU matrix on the custom
`fp16io-fp16-webgpu` preset (`fp16_iofp32_fp16out` encoder + `fp16` decoder):

| Case                                      |             Total |           RTFx | Steps | KV         | Downloads | Parity                  |
| ----------------------------------------- | ----------------: | -------------: | ----: | ---------- | --------: | ----------------------- |
| EN greedy GPU-KV, 30s                     |         1167.54ms |       25.6131x |    49 | gpu-buffer |         0 | coherent                |
| EN stable beam 2, 30s                     |       10914.095ms |        2.7400x |    98 | cpu        |         0 | oracle                  |
| EN batched beam 2, 30s                    |         8779.44ms |        3.4062x |    49 | cpu        |         0 | exact tokens            |
| TR auto stable beam 2, 18s                |        16593.42ms |        1.1234x |   158 | cpu        |         0 | oracle                  |
| TR auto batched beam 2, 18s               |        14348.08ms |        1.2992x |    79 | cpu        |         0 | exact tokens            |
| EN timestamped stable/batched beam 2, 10s | 3930.49/2802.21ms | 2.5453/3.5702x | 40/20 | cpu        |         0 | exact tokens + 17 words |

The harness reports `check` because it deliberately caps generated tokens;
there were no inference or page errors. GPU-KV greedy retained zero GPU
downloads.

## Timestamp boundary result

The original alignment export omitted the causal self-attention mask, so
teacher-forced rows could see future text tokens. The runtime also omitted
Whisper's `<|notimestamps|>` anchor row, re-softmaxed already-normalized
`decoder_align` weights, and applied a generic long-word clip to verified DTW
boundaries. Those independent boundaries explain the former zero anchor and
the later-span drift.

The corrected local export is at
`N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r2`. It contains all
four FP16 graphs, co-located `.onnx.data` files, the causal alignment marker,
and tokenizer/config files. ONNX checker and CPU ONNX Runtime loading passed
for every graph. A direct PyTorch/ONNX alignment check found maximum absolute
output difference `5.2e-4` and mean absolute difference `6.9e-7`; the graph
rows are post-softmax weights with row sums of 1.0 before short-clip cropping.

The runtime now uses the reference sequence
`[SOT, language, task, no_timestamps, text..., EOT]`, extracts the anchor plus
text rows, crops and renormalizes padded attention rows, and preserves long
non-punctuated boundaries from verified forced-DTW. The same row contract is
used by the merged path and the portable WhisperX runner. The official
reference constructs the same sequence and slices the same anchor-containing
matrix; see [Whisper `timing.py`](https://github.com/openai/whisper/blob/main/whisper/timing.py).

With only the corrected external-data `decoder_align` graph temporarily
installed in the actual headless Chrome/WebGPU harness, the final 10.004s JFK
measurement was:

| Measurement        |                   Value |
| ------------------ | ----------------------: |
| Warm WebGPU total  |             `776.315ms` |
| RTFx               |              `12.8869x` |
| Encode / decode    | `184.105ms / 471.635ms` |
| Decoder steps      |                    `20` |
| First word         |         `In 2.32–2.84s` |
| Long-span check    |       `have 7.30–8.88s` |
| Word count         |                    `17` |
| GPU KV / downloads |        `gpu-buffer / 0` |
| Warnings           |                    none |

The cached faster-whisper CUDA reference produced the same transcript and
began `In` at `2.16–2.84s`; its long `have` span was `7.46–8.86s`. Browser
punctuation collation differs (`role` plus `...` versus `role...`), but the
alignment remains anchored and the long non-punctuated span is preserved.

The checked-in change is runtime/exporter-contract code, not a replacement
model binary. The browser harness files were restored byte-for-byte after
each probe. The published 4-graph model still must be re-exported and
validated per precision variant before a remote preset can claim this
artifact-level timestamp behavior. No model-hosting update was performed.

## Follow-up: selected-head raw-logit alignment contract

The legacy post-softmax averaged graph was not sufficient for short-clip
parity: it discarded the selected-head axis before the runtime cropped the
30-second encoder window. The exporter and runtime now support the reference
contract directly. New `decoder_align` graphs return
`[batch, selected_head, target_sequence, source_frames]` raw cross-attention
logits and declare `attention_values: "logits"` plus
`attention_layout: "selected_heads"` in `manifest.json`. The runtime then
softmaxes each head after the frame crop, normalizes all teacher-forced rows,
median-filters, averages heads, and selects the no-timestamps anchor plus text
rows. Legacy `[batch, target_sequence, source_frames]` post-softmax graphs
remain supported through the manifest-declared compatibility path.

Two complete local exports were validated without changing the published
remote artifact:

- `N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r3`: FP16 raw-logit
  alignment output. ONNX checker, external-data path validation, and ORT
  loading passed for all four graphs.
- `N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r4`: same contract
  with the alignment q/k score accumulation exported as float32. Its
  `decoder_align` output is `float32`; the other graph boundaries remain
  unchanged. Random-input PyTorch/ORT alignment checks passed for both
  variants (r4 max absolute difference `0.01616`, mean absolute difference
  `0.00138`).

Independent Chrome/WebGPU A/B probes installed each local graph only
temporarily and restored the public model files afterward. Both r3 and r4
preserved the transcript and, before the punctuation-collation fix, produced
the first word as `In 2.32–2.84s`; the float32 alignment score path did not move
that boundary. The local PyTorch forced-alignment reference produced a first
text boundary near `2.82s`. The multi-character punctuation fix later moved the
complete r4 browser output to `In 2.12–2.84s`, close to the faster-whisper
`2.16–2.84s` reference, without changing the raw DTW anchor. The remaining
hidden-state variance is still in the browser encoder path, so do not publish
r3/r4 or claim exact native first-word parity until that numerical boundary is
independently characterized.

The TypeScript DTW recurrence now also matches OpenAI Whisper's float32 cost
buffer and explicit zero-border backtrace. This is covered by the existing
alignment tests and does not change the public timestamp-array shape.

## Merged-decoder alignment boundary

The merged-decoder path had a separate, untested alignment implementation that
had drifted from the corrected split-graph contract. It previously omitted
`<|notimestamps|>` from the teacher-forced sequence, hard-coded the prompt
rows, read attention rows beginning at row zero, and inferred the frame count by
halving encoder hidden-state positions. The runtime now uses
`[SOT, language, task, no_timestamps, text..., EOS]`, derives the prompt length
from that sequence, reads the causal logit row that predicts each text token,
retains the full teacher-forced attention rows for token-axis normalization,
selects the no-timestamps anchor plus text rows for DTW, and crops/renormalizes
the attention frame axis to the input audio duration. The merged path now uses
the same full-row normalization contract as the split-graph WebGPU path,
including the full-frame row stride when an audio-duration crop is requested.

This boundary is covered by
`tests/whisper-merged-alignment.test.ts`. No local merged-decoder artifact with
exported `cross_attentions.*` outputs is currently available, so the remaining
end-to-end claim is artifact-gated rather than presented as a browser benchmark.

The ordinary merged decoder is now exercised end-to-end by the portable
`tests/whisper-onnx-smoke.test.ts` fixture (set
`WHISPER_MERGED_FIXTURE_DIR` on Windows). A direct local run against
`N:\github\huggingface\whisper-small-dsntt1-tr-onnx` and the first six seconds
of `tr-tdk-18s.wav` produced coherent Turkish text, nine words, and no warnings
in Node/WASM. Before the duration-boundary fix, generated padded-window
timestamps reached 8.0 seconds; after it, both the maximum word end and maximum
segment end were exactly 6.0 seconds. This validates merged decode plus the
timestamp-token fallback, not attention-DTW alignment.

## Current performance reference

The mel benchmark reports both contracts explicitly (`n_mels=128`, five runs on
the local Node host). A fresh sample measured exact 400-point default at
`174.9ms` for 30s audio versus experimental 512-point at `56.0ms` (about 3.1x
faster). The latter remains opt-in because the speedup changes the model's
frequency-bin input contract; the earlier `177.3ms`/`49.0ms` sample remains a
valid historical measurement, not a guaranteed throughput floor.

Independent local faster-whisper CPU/int8 on the same 29.904s JFK fixture,
with three warmed measurements per beam, produced:

| Backend / beam              | Median inference |     RTFx | Text      |
| --------------------------- | ---------------: | -------: | --------- |
| faster-whisper CPU int8 / 1 |        `16.014s` | `1.867x` | identical |
| faster-whisper CPU int8 / 2 |        `16.055s` | `1.863x` | identical |
| faster-whisper CPU int8 / 5 |        `16.259s` | `1.839x` | identical |

The same cached faster-whisper CUDA model was also measured on the 10.004s JFK
clip: beam 1 `431.054ms` / `23.2082x` RTFx, beam 2 `495.882ms` / `20.1742x`,
and beam 5 `484.535ms` / `20.6466x`. These are native GPU inference timings,
not browser session-load timings.

## Follow-up: artifact capability audit and merged fallback warning (2026-08-25)

The local ONNX audit now has a Whisper-specific contract mode:

```text
node tools/model-debugging/scripts/node-audit-onnx-artifact.mjs \
  --model-dir <artifact-dir> --recursive --whisper-contract
```

It reports the layout, the explicit `alignment_export.causal_self_attention`
marker, and every merged decoder's `cross_attentions.*` output names. Optional
`--require-causal-alignment` and `--require-merged-cross-attention` flags fail
the audit when a publish candidate does not meet those claims. This keeps an
artifact filename or a generic `decoder_model_merged.onnx` from being treated
as proof of attention alignment.

The audit was run against the local artifacts currently available:

| Artifact                                         | Graph result      | Timestamp capability                                                     |
| ------------------------------------------------ | ----------------- | ------------------------------------------------------------------------ |
| `whisper-large-v3-turbo-causal-fp16-20260825-r4` | 4/4 graphs loaded | `splitgraph-causal-attention-dtw`                                        |
| `base-dsntt1-tr/onnx`                            | 8/8 graphs loaded | `merged-generated-timestamp-fallback`; zero `cross_attentions.*` outputs |

The merged executor now emits a recoverable warning when alignment heads or
`cross_attentions.*` are absent, while preserving timestamp-token interpolation
and duration clipping. Focused tests cover the warning and the artifact
classification. The local base merged graph therefore validates ordinary
merged decode and bounded fallback timestamps, but it does not close the
merged attention-DTW requirement.

The opt-in merged timestamp smoke used `tr-tdk-18s.wav` (18.641 s) with the
local base model and 96 new-token limit. Node/WASM produced 29 words in
`59.291s`, with maximum word and segment end both `18.641s`, and emitted
`whisper.decoder-cross-attention-unavailable`. The text is useful as a runtime
smoke result, not as an attention-alignment quality claim.

Final 10.004s WebGPU timestamped probes with the corrected local alignment
graph retained exact stable/batched text and word parity:

| Browser mode   |       Total |       RTFx | Decoder steps | Tokens |
| -------------- | ----------: | ---------: | ------------: | -----: |
| Greedy GPU-KV  | `776.315ms` | `12.8869x` |          `20` |   `18` |
| Stable beam 2  | `3930.49ms` |  `2.5453x` |          `40` |   `18` |
| Batched beam 2 | `2802.21ms` |  `3.5702x` |          `20` |   `18` |

The final batched beam run was `1.40x` faster than stable beam 2, used half
the decoder steps, kept CPU KV for correctness, and matched the stable words
exactly. The greedy run kept GPU KV with zero GPU downloads. The harness
reported `check` because its generated-token cap is deliberate.

After the immutable-cache sharing change, warmed 29.904s English beam 2
repeats improved while retaining exact stable/batched text parity:

| Browser mode   |        Total |      RTFx | Decoder steps | Change vs prior reference |
| -------------- | -----------: | --------: | ------------: | ------------------------: |
| Stable beam 2  | `9571.865ms` | `3.1242x` |          `98` |            `12.3%` faster |
| Batched beam 2 |  `7850.17ms` | `3.8094x` |          `49` |            `10.6%` faster |

The run kept CPU KV for beam, zero GPU downloads, and identical generated
text. The cache-sharing contract is therefore a measured optimization, not a
change to beam ranking or output semantics.

The 30s greedy GPU-KV repeat reached `1080.68ms` / `27.6718x`, with 49
decoder steps and zero GPU downloads. The earlier 30s beam matrix remains the
long-audio parity reference above; one repeat of stable beam 2 did not produce
a result within 166s and was excluded rather than treated as a performance
claim.

The current exact-mel follow-up on the same 30s fixture produced these warm
measurements with the restored local harness:

| Browser mode   |        Total |       RTFx | Decoder steps | Token parity |
| -------------- | -----------: | ---------: | ------------: | ------------ |
| Greedy GPU-KV  | `1215.745ms` | `24.5975x` |          `49` | baseline     |
| Stable beam 2  |  `8818.83ms` |  `3.3910x` |          `98` | oracle       |
| Batched beam 2 |  `7455.44ms` |  `4.0111x` |          `49` | exact stable |

The second measurement in each mode was `19.935x`, `3.2988x`, and `3.8424x`
respectively; stable and batched beam produced identical token sequences.
These remain opt-in/diagnostic beam modes, while GPU-KV greedy is the fast
path.

### Live optimization audit (2026-08-25)

The current browser harness was used for an A/B probe against the same local
`fp16io-fp16-webgpu` artifact and 29.904s JFK fixture.

- Enabling `decoderGraphCapture=1` originally failed during ORT session
  creation: `all compute graph nodes have not been partitioned to the
WebGpuExecutionProvider`.
- The runtime now retries that opt-in request without graph capture and emits
  `whisper.decoder-step-graph-capture-fallback`. A headless Chrome/WebGPU run
  completed with a coherent transcript, `27.5352x` RTFx, GPU-KV, and zero GPU
  downloads. This is a compatibility fallback, not a graph-capture speedup.
- `decoderFreeDimensionOverrides` was measured as an opt-in diagnostic. Actual
  totals were `921.94ms`, `1033.535ms`, and `1095.44ms` for the override versus
  `1085.835ms`, `1090.55ms`, and `1177.945ms` for paired baseline runs. The
  variance is large enough that the override is not promoted or enabled by
  default.
- The local native references are faster-whisper CPU/int8 at about `1.84x` to
  `1.87x` RTFx on the 30s fixture and faster-whisper CUDA float16 at about
  `20.17x` to `23.21x` RTFx on the 10s fixture. `N:\models\whisper-cpp`
  contains GGML weights but no runnable `whisper-cli`/`main` executable, so no
  whisper.cpp timing is claimed.

- A fresh 10.004s faster-whisper CPU/int8 check on the same JFK clip measured
  `7.954s`, `8.004s`, and `8.071s` for beams 1, 2, and 5 (`1.258x`, `1.250x`,
  and `1.240x` RTFx). All three returned the same transcript and first-word
  span `2.16–2.84s`. WhisperX's locally runnable no-align path produced only
  a segment-level result (`2.613–10.021s`); no compatible cached English word
  alignment model was available, so it is not used as a word-timestamp or
  performance claim.

- The batched decoder-step path now returns zero-copy typed-array views when
  splitting present KV outputs back into per-beam caches. Input packing still
  clones into fresh batch storage, so the ORT input-safety boundary is
  unchanged. In the same 30s browser harness, the measured
  `decoderStepKvMergeMs` bucket fell from `12–47ms` per run before this change
  to `3–5ms` after it; the post-change beam-2 and beam-5 runs retained exact
  stable/batched token parity. A second direct-pack follow-up reduced the
  `decoderStepFeedBuildMs` total over 49 batched calls from about
  `2.9–3.1s` to `1.57s` for beam 2 and from `6.94–6.96s` to `3.57–3.63s`
  for beam 5. Total WebGPU time remains variable, so this is kept inside the
  existing opt-in `experimentalBatchedBeam` path.

## Continuation audit (2026-08-25)

The remaining timestamp discrepancy was isolated with a complete local r4
graph set and an independent CPU ONNX Runtime reference. The r4 artifacts
were temporarily installed into both the decoder and encoder folders of the
browser harness, then restored from backups after the probes.

- The r4 CPU reference used the same 128-bin TypeScript-compatible Whisper mel
  contract and returned the raw DTW sequence beginning `[0.00, 2.82, 3.00,
3.48, ...]` for the first text boundary. The Python/OpenAI mel output and
  the TypeScript mel output matched within float32 roundoff (same frame count,
  extrema, and first-frame values).
- The complete r4 WebGPU browser run kept the transcript exact, GPU-KV, and
  zero GPU downloads, but returned `In 2.32–2.84s`. Replacing the public
  optimized encoder with the r4 encoder did not move that boundary.
- A split-backend control with the r4 WebGPU encoder and r4 decoder/alignment
  graph on WASM also returned `2.32–2.84s`. This rules out the WebGPU
  `decoder_align` execution provider and the DTW row-selection implementation
  as the source of the difference; the sensitivity is in the WebGPU encoder
  numerical path feeding forced alignment. The r4 WASM fp16 encoder control
  was not viable (`std::bad_alloc`), so it is not a performance or correctness
  oracle.
- The optional Wav2Vec2 refinement on the complete r4 set returned
  `In 2.786–2.866s`, close to the CPU forced-alignment boundary. No timestamp
  fudge was added: a model export with a verified fp32-accumulation encoder,
  or an explicit alignment-only CPU reference path, is still required before
  claiming native first-word parity.

The portable `tests/smoke/whisperx-runner.mjs` was tightened as part of this
audit. It now locates RIFF `fmt `/`data` chunks instead of assuming a 44-byte
header, preserves declared fp16/fp32 KV dtypes, and uses the same encoder and
alignment boundary casts as the main executor. New tests cover metadata-chunk
WAV parsing and fp16 KV preservation. A real fp32 splitgraph smoke completed
with the exact `10.0043125s` audio duration and the expected English
transcript.

The public harness was restored after validation. Original hashes remain:
legacy `fp16/decoder_align.onnx` `2B730AE7...`, its data file
`94AEB6AB...`, `fp16/manifest.json` `309FD78F...`, and the optimized encoder
`20EDF7D5...`.

## Follow-up: FP16 feature boundary and raw first-word anchor (2026-08-25)

An independent browser encoder probe found one additional runtime boundary bug
before the encoder: the JavaScript float32-to-float16 helper used an incorrect
shift for float32 subnormals. Padded Whisper mel frames contain values in this
range, so the old conversion could turn values around `1e-5` into `+/-2`.
`float32ToFloat16Bits()` now uses the IEEE-754 subnormal shift and
round-to-nearest-even, reads source bits through one typed-array view, and has a
regression test for normal values, subnormals, and ties. The browser probe
reported zero differing FP16 bit patterns against NumPy after the fix.

The corrected cast did not eliminate the remaining hidden-state difference:
the same browser mel and r4 encoder input produced WebGPU-versus-CPU-ONNX
cosine `0.9620`, while CPU-ONNX-versus-local-PyTorch-CUDA was `0.99948`.
This isolates the residual variance to the WebGPU FP16 encoder execution path,
not the mel/FP16 input boundary, decoder alignment graph, or DTW row selection.

The complete corrected r4 browser run still retained exact transcript and word
parity:

| Case                          |        Total |       RTFx | Steps | KV         | First word      |
| ----------------------------- | -----------: | ---------: | ----: | ---------- | --------------- |
| EN timestamped greedy         |  `796.415ms` | `12.5617x` |    20 | GPU buffer | `In 2.32-2.84s` |
| EN timestamped stable beam 2  | `3949.825ms` |  `2.5328x` |    40 | CPU        | `In 2.32-2.84s` |
| EN timestamped batched beam 2 |  `2831.57ms` |  `3.5331x` |    20 | CPU        | `In 2.32-2.84s` |

The raw WebGPU forced-alignment probe returned the anchor-containing DTW
sequence `[0.00, 2.84, 2.98, 3.48, ...]`. The first output word therefore
starts at the preserved anchor `0.00` before duration postprocessing. The
pre-fix `2.32` start was the intentional long-leading-pause clamp (`end -
2*median_word_duration`) also used by the local OpenAI Whisper timing
reference. This is an output-semantics clarification, not a timestamp offset
to add.

## Follow-up: multi-character punctuation collation (2026-08-25)

The tokenizer can emit an appended punctuation run such as `...` as one BPE
token. The previous punctuation merge checked membership of the entire token
against the single-character punctuation string, so the run stayed as a
separate word (`role` plus `...`). The merge now accepts a whitespace-free
punctuation-only token when every character belongs to the configured appended
punctuation set. A focused DTW regression covers the merged text, token IDs,
source indices, and end boundary.

The corrected r4 browser timestamp matrix produced one `role...` word and moved
the first word to `2.12–2.84s`, close to the faster-whisper `2.16–2.84s`
reference. Stable and batched beam 2 returned exact token and word parity:

| Case                          |        Total |     RTFx | Steps | KV         | First word      |
| ----------------------------- | -----------: | -------: | ----: | ---------- | --------------- |
| EN timestamped greedy         |  `811.245ms` | `12.332` |    20 | GPU buffer | `In 2.12-2.84s` |
| EN timestamped stable beam 2  |  `4030.81ms` |  `2.482` |    40 | CPU        | `In 2.12-2.84s` |
| EN timestamped batched beam 2 | `2863.865ms` | `3.4933` |    20 | CPU        | `In 2.12-2.84s` |

These are correctness-validation runs, not a new performance baseline; repeat
warmed samples are still required for timing claims. The remaining small
first-word difference is within one 20 ms timestamp quantum plus encoder
numeric variance and does not justify a timestamp fudge.

## Follow-up: synchronization diagnostics and encoder quantization (2026-08-25)

The remaining encoder-boundary probes were completed against the saved local
r4 graph set. Forcing encoder output to CPU or re-wrapping the GPU buffer kept
the same `In 2.12–2.84s` first-word span, while adding the explicit GPU drain
added roughly 195 ms of profiling overhead without changing the words. The
drain path had been dropping the already-downloaded CPU view when it re-wrapped
the GPU buffer; the runtime now retains that view so decoder-align probes stay
on the real alignment path instead of silently falling back to generated
timestamp interpolation. The corrected flush probe returned 16 words,
`In 2.12–2.84s`, GPU-KV, zero GPU downloads, and no warnings.

The locally available q8 encoder was also tested with the fp32 decoder in the
same headless WebGPU harness. It preserved the English transcript and GPU-KV
output, but took `47,745.245ms` for the warmed 10.004s clip, including
`47,395.12ms` in the encoder (`0.2095x` RTFx). Quantized encoder operators are
therefore not a viable WebGPU optimization with the current ORT-WebGPU build;
the fp16 split encoder remains the production candidate.

## Follow-up: reproducible r5 browser validation and FP32 reference A/B (2026-08-25)

The fresh offline export
`N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5` was validated in the
actual headless Chrome/WebGPU harness by temporarily exposing the local artifact
through a junction. The junction and harness changes were removed after the
probe. The warmed 10.004s JFK run used r5 for all four graphs, greedy GPU-KV,
timestamped decoding, and returned the exact baseline transcript plus 16 words
with no warnings:

| Case           |   Total |      RTFx | Encoder | Decoder | KV           | Downloads | First word      |
| -------------- | ------: | --------: | ------: | ------: | ------------ | --------: | --------------- |
| r5 FP16 WebGPU | `875ms` | `11.555x` | `191ms` | `482ms` | `gpu-buffer` |       `0` | `In 2.12-2.84s` |

The native runner now loads the same FP16 artifact without a manual process
shim. Its alignment result is also stable: `In 2.12-2.84s`, `long 2.98-3.44s`,
and `role... 9.68-9.98s`. The runner fix was twofold: the decoder-align output
tensor no longer shadows the encoder input tensor, and Node 26's native
`Float16Array` is hidden before importing the installed ORT binding, which
currently expects the portable `Uint16Array` FP16 representation.

For the quality/performance boundary, the existing public full-FP32 WebGPU
preset was warmed on the same 10.004s clip. It retained the exact transcript
and timestamp-token sequence but measured `1.44s` / `6.972x`, including `923ms`
of encoder time, versus r5's `875ms` / `11.555x`. Full FP32 is therefore a useful
reference-quality diagnostic, not a practical default or an alignment-only
replacement without a separate artifact and an explicit cost contract.

The r5 result does not move the first-word boundary or eliminate the previously
measured WebGPU-versus-CPU encoder hidden-state variance. No timestamp offset or
fudge is introduced. The remaining exact-native-parity boundary is still an
encoder export/runtime question: either a verified FP32-accumulation browser
encoder or an explicit, opt-in CPU alignment reference is needed before making
that claim.

## Follow-up: causal FP32 browser reference (2026-08-25)

The fresh offline export
`N:\models\whisper-large-v3-turbo-causal-fp32-20260825-r1` was then run through
the same local Chrome/WebGPU harness. The temporary junction and result probe
were removed after the run. With all four r1 graphs, greedy GPU-KV, and
timestamped decoding, the browser returned the exact baseline transcript and
token sequence plus all 16 native word anchors:

| Case                  |   Total |     RTFx |  Encoder | Decoder | KV           | Downloads | First word      |
| --------------------- | ------: | -------: | -------: | ------: | ------------ | --------: | --------------- |
| r1 FP32 WebGPU causal | `4.82s` | `2.075x` | `1.493s` | `329ms` | `gpu-buffer` |       `0` | `In 2.12-2.84s` |

The browser/native anchors agree at the checked boundaries: `In 2.12-2.84s`,
`long 2.98-3.44s`, `world, 4.30-5.38s`, and `role... 9.68-9.98s`. This closes
the causal FP32 artifact's timestamp parity boundary and isolates the remaining
quality difference in r5 to the FP16 encoder path. r1 is a correctness
reference, not a default performance target: its warmed browser encoder is
about 7.8x slower than r5's `191ms` encoder on this clip. Neither local export
was copied into a public model folder or uploaded.

## Follow-up: shared encoder KV for batched beams (2026-08-25)

The batched decoder profile identified a safe memory-bandwidth waste for the
validated dynamic-token graph: every active beam was repacking the same
immutable encoder key/value cache on every `decoder_step`. The core read-only
cache contract shares the initial cache across sibling beams. For that graph
family the WebGPU split-graph adapter passes encoder KV with batch `1`, which
ONNX attention broadcasts across the active token batch, while decoder
self-attention KV remains fully batched. The later fixed-token r5 export showed
that an accepted shape is not sufficient evidence of correct broadcast; the
executor now gates this optimization by graph capability and uses a cached
materialized fallback when needed.

An independent CPU ONNX probe accepted the `[1, heads, frames, head_dim]`
encoder-KV input with `[2, 1]` token inputs. The browser harness then validated
the optimized path against the unmodified public model:

| Case                                  |         Total |      RTFx | Steps | Token/word parity | Downloads |
| ------------------------------------- | ------------: | --------: | ----: | ----------------- | --------: |
| EN timestamped batched beam 2, before |   `2928.82ms` | `3.4158x` |    20 | baseline          |         0 |
| EN timestamped batched beam 2, after  |   `2059.86ms` | `4.8568x` |    20 | exact vs stable   |         0 |
| EN beam 5, stable                     | `22919.805ms` | `1.3047x` |   245 | oracle            |         0 |
| EN beam 5, batched                    |  `5038.375ms` | `5.9353x` |    49 | exact vs stable   |         0 |

For beam 2, `decoderStepFeedBuildMs` fell from `638.29ms` to `335.045ms` and
CPU tensor inputs fell from `682` to `342`. The optimization remains behind
`experimentalBatchedBeam`; stable CPU-KV remains the correctness oracle and
GPU-KV remains greedy-only.

## Follow-up: explicit alignment gating and beam candidate pruning (2026-08-25)

The public `ysdede/whisper-large-v3-turbo-onnx-4graph` manifest still has no
valid `alignment_export.causal_self_attention` marker. The executor previously
treated only an explicit `false` as legacy, which left missing metadata
eligible for `decoder_align`. The guard now requires an explicit `true`; a
missing or malformed marker emits the recoverable legacy warning and uses
generated timestamp interpolation. A live browser run against the public
manifest produced both legacy warning codes and the expected `In 0.00–0.20s`
interpolated first word. The local r4 manifest explicitly declares `true` and
remains eligible for the causal DTW path; no timestamp offset or fudge was
introduced.

Beam ranking also now selects only the top `beamWidth` token IDs from each
source beam before materializing score objects. This is equivalent to the
previous exhaustive global top-k insertion because no lower-ranked token from
one source beam can enter the global top-k set. Equal logits retain ascending
token-ID order. A Node microbenchmark with five beams and the 51,866-token
vocabulary measured `9.738ms → 6.260ms` per ranking call (`35.7%` lower), and
1,000 randomized comparisons matched the exhaustive implementation exactly.
The optimization is CPU-side; the real WebGPU matrix remains GPU-dominated, so
no unsupported end-to-end speedup is claimed.

For the same warmed 29.9043s JFK clip, local `faster-whisper` measured on CUDA
FP16 at `0.914s` inference (`32.73x` RTFx) after a `1.925s` warm run. The
existing independent WebGPU greedy reference is `25.6993x` RTFx; these are
reference points rather than interchangeable implementations. No local
OpenAI Whisper checkpoint or whisper.cpp CLI was available, so those baselines
were not fabricated or downloaded.

## Follow-up: scalar decoder KV bridge (2026-08-25)

The callback-based split-graph scalar path had been copying each CPU KV cache
twice: once while normalizing raw callback values into step-input tensors, then
again inside the generic decoder-step runner. The adapter now marks its freshly
owned, correctly named tensors as prepared, so the generic runner forwards them
without a second copy. All other callers retain the defensive cross-session
clone, including CPU tensors on the GPU-KV fallback path. Token-ID tensor
construction also no longer creates a temporary array through `map(BigInt)`.

A same-build synthetic boundary benchmark over twenty large-cache iterations
measured the defensive-versus-prepared bridge at `21.85ms → 11.18ms`,
`10.0M → 5.0M` copied elements, and `60 → 40` tensor constructions. This is a
feed-boundary measurement, not an end-to-end model claim. The focused fp16 KV
test now asserts that prepared tensors are forwarded by identity and that only
`input_ids` is constructed inside the generic step runner.

Independent headless Chrome/WebGPU validation against the current local public
model kept exact greedy/stable/batched transcript parity and zero GPU
downloads. Warm measurement runs on the 29.9043s JFK fixture returned:

The unqualified rows below use the `fp16io` encoder with the `fp32` decoder;
rows labeled `full-fp16` use the full-FP16 decoder.

| Case                       |    Total |     RTFx | Step calls | Feed build | KV         |
| -------------------------- | -------: | -------: | ---------: | ---------: | ---------- |
| EN greedy GPU-KV           |  `1.38s` |  `21.77` |         49 |          — | GPU buffer |
| EN full-fp16 greedy GPU-KV |  `1.39s` | `21.699` |         49 |          — | GPU buffer |
| EN full-fp16 stable beam 2 |  `8.42s` |  `3.556` |         98 |    `1.54s` | CPU        |
| EN stable beam 2           | `12.31s` |  `2.431` |         98 |    `2.97s` | CPU        |
| EN batched beam 2          |  `6.91s` |  `4.332` |         49 |    `1.53s` | CPU        |

The stable beam run reported only `3ms` of generic-step tensor-clone time;
the remaining scalar feed-build time is the single required raw-cache copy.
The batched run remains governed by its separate batch-packing path. GPU-KV
greedy remains the fast path, stable CPU-KV beam remains the correctness
oracle, and batched beam remains opt-in.

## Follow-up: reproducible corrected fp16 export (2026-08-25)

A fresh offline export from the cached local Whisper checkpoint produced
`N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5`. The exporter ran
with the machine's CUDA-enabled Python 3.11 environment and emitted all four
FP16 graphs with co-located external data. Path-based ONNX checking, CPU ORT
loading, and the Whisper contract audit passed for all four graphs:
`causal_self_attention=true`, raw-logit alignment, and six selected heads with
alignment output shape `[1, 6, 20, 1500]`.

The independent Node validator compared the r5 FP16 graphs with the current
public FP32 baseline on three small English/Turkish fixtures at 16 generated
tokens. All three had exact `16/16` token parity, matching text, monotonic
alignment, and the expected raw-logit alignment contract. Node 26's native
`Float16Array` is currently rejected by the installed `onnxruntime-node`
binding; the validator therefore used a process-local compatibility shim that
restores the established `Uint16Array` FP16 representation. This is a native
validator/runtime limitation, not a browser WebGPU result.

The r5 artifact remains local and was not copied into the public model folders
or uploaded to a remote repository. The corrected graph still needs a fresh
browser run for each published precision pairing before remote promotion.

## Follow-up: mixed-precision encoder probe and external-data compatibility (2026-08-25)

A controlled local graph rewrite was tested against r5 with FP32
`LayerNormalization` and `Softmax` islands surrounded by FP16 casts. CPU ORT
gave a small hidden-state improvement toward the FP32 reference (cosine
`0.999989867` versus r5 `0.999985397` on the JFK mel fixture), and an
independent browser run preserved the exact 16-word anchors:
`In 2.12–2.84s`, `long 2.98–3.44s`, `world, 4.30–5.38s`, and
`role... 9.68–9.98s`. It is not a production candidate: the mixed encoder
measured roughly `568–1,420ms` on the 10s WebGPU fixture versus r5's warmed
`191ms`, with no observable timestamp improvement. The temporary artifact was
kept out of public model folders and remote hosting.

The runtime also now forwards every manifest-declared external-data shard to
ORT instead of selecting only the first entry. The one-file `.onnx.data`
layout used by the current Whisper exports is unchanged, while multi-shard
exports now remain loadable in browser and Node session paths. A focused
artifact test covers the complete shard list.

## Follow-up: committed-runtime browser smoke (2026-08-25)

After the shard bridge fix was pushed, an independent browser matrix run loaded
the current local `fp16io-fp16-webgpu` public-model variant from the sibling
harness. The 29.904-second JFK fixture produced the expected transcript prefix
(`In the long history of the world, only a few generations ...`) at `24.1657x`
RTFx, with `1,237.47ms` total time, `183.49ms` encoder time, `812.02ms`
decoder time, 49 decoder steps, GPU-KV location `gpu-buffer`, and zero GPU
downloads. The harness status was `check` only because its 50-token cap
truncated the longer text oracle. The model emitted the documented recoverable
`whisper.decoder-align-legacy` warning because the currently published local
alignment graph lacks the causal-self-attention export marker; timestamped
artifact promotion remains gated on re-exporting that graph.

## Remaining boundary

The local split-graph timestamp contract is now fixed and independently
validated: the no-timestamps anchor, model-specific fallback ID, post-softmax
attention crop/renormalization, multi-character punctuation collation, long
non-punctuated DTW spans, and stable versus batched beam word parity all pass
focused tests and the corrected r5 browser probe. The corrected graph must
still be regenerated for each published precision variant and validated on the
remote model before the default preset can claim artifact-level timestamp
parity. Merged-decoder end-to-end validation still needs a timestamped merged
graph with `cross_attentions.*` outputs and a broad English/Turkish reference
fixture.

The FireRedASR2S source tree is present at `N:\github\ysdede\FireRedASR2S`,
but no ASR2 checkpoint, `cmvn.ark`, or converted runtime artifact is available
locally. The cached Qwen material is an unrelated text model, so the package's
FireRed VAD support and Qwen ASR documentation remain artifact-gated rather
than being expanded with an unverified runtime.

Keep `experimentalBatchedBeam` opt-in, stable CPU-KV beam as the correctness
oracle, and GPU-KV greedy-only until broader model/reference coverage closes
these boundaries.

## Follow-up: HF beam oracle and score-based stopping (2026-08-25)

The local HF checkpoint now has a committed deterministic reference at
`tools/data/results/whisper/whisper-large-v3-turbo-jfk2-beams.json`. It records
the JFK2 WAV SHA-256, the complete prompt `[50258, 50259, 50360, 50364]`, EOS,
and exact token/text outputs for beam 1, 2, and 5. The generator can reproduce
it with `--beam-sizes 2,5 --length-penalty 1`; the reference explicitly fixes
`do_sample=false`, `length_penalty=1`, and `early_stopping=false`.

The native r5 splitgraph runner exposed a real beam-5 boundary: the old loop
stopped as soon as five EOS candidates existed and returned a sequence missing
the comma token after `so`. The beam-2 output was unaffected. The core loop
now retains finished candidates and applies a score-based stop heuristic after
the patience target is filled, matching the default Transformers behavior
while keeping explicit `patience` as the completion target. The runner also
now exposes programmatic `tokenSequences` without changing CLI output files.

Against `N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5`, the native
oracle passed exact token and text parity for beam 2 and beam 5. The focused
Vitest fixture/provenance checks and the artifact-gated smoke are:

```text
npx vitest run tests/whisper-beam-reference.test.ts
node tests/smoke/whisper-beam-reference.mjs --model N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5
```

The fast path boundaries remain unchanged: GPU-KV is greedy-only, stable
CPU-KV beam is the correctness oracle, and batched beam remains opt-in.

## Follow-up: fixed-token r5 batched-KV compatibility (2026-08-25)

The r5 `decoder_step.onnx` graph accepts batch-shaped token inputs but does not
correctly broadcast a batch-one encoder KV tensor. A direct CPU ORT probe showed
that the shared `[1, 20, 1500, 64]` cache changed the first-step top token from
`370` to `11`/other values for different rows; fully materializing the same
encoder cache to `[5, 20, 1500, 64]` matched the scalar logits exactly. The
older public graph has a dynamic token axis and passed the broadcast probe.

The executor now gates encoder-KV broadcast on that declared dynamic token
axis, defaults unknown graph shapes to the safe materialized path, and retains
the materialized immutable encoder tensors per active batch shape for the rest
of the decode. Decoder self-attention KV remains fully batched and changing
beam counts still rebuild the required shape.

Independent Chrome/WebGPU validation against the local r5 junction produced
exact transcript parity with no warnings or GPU downloads:

| Case | Total | RTFx | Step calls | KV | Exact |
| ---- | ----: | ---: | ---------: | -- | :---: |
| EN stable beam 5 | `18.605s` | `1.6073` | 245 | CPU | oracle |
| EN batched beam 5 | `7.110s` | `4.2062` | 49 | CPU | yes |

The fixed-token safe path's batch feed-build time fell from about `3.42s` to
`0.18s` over the 49-step decode after tensor reuse. This remains an opt-in
optimization; the artifact capability gate prevents a silent wrong-transcript
promotion.

## Follow-up: FP16/FP32 timestamp boundary audit (2026-08-25)

A fresh parity probe used an exact FFmpeg-decoded WAV (`160069` samples at
16 kHz) so browser media decoding could not be confused with model variance.
The native r4, r5, and causal FP32 r1 runners all placed the checked boundary
at `world, 4.30-5.38s`. The r5 FP16 WebGPU path placed `long` at `2.98-3.44s`
and `world,` at `4.30-5.40s`; the browser OGG path also moved `long` one frame
later, while the exact WAV removed that media-decoder contribution.

The full FP32 r1 WebGPU control returned all checked native anchors, including
`world, 4.30-5.38s`. Repeating the r5 control with encoder output forced to
CPU, an explicit GPU flush, a GPU drain, and a buffer re-wrap did not move the
`world,` boundary. This keeps the diagnosis narrow: the residual is one
20-millisecond frame from FP16 WebGPU encoder execution feeding the alignment
graph, not a prompt row, crop, DTW, punctuation, or global offset error.

No timestamp offset or one-frame correction is justified. The production
contract remains frame-quantized timestamps with exact native parity gated on
an FP32-accumulation encoder or an explicitly selected reference artifact.
The existing `maybeCastEncoderHiddenStates` boundary support is sufficient for
an alignment graph with a different declared input dtype; a separate reference
encoder is intentionally not run by the fast default because it would duplicate
the dominant encoder pass.

The same local 10-second benchmark also records the performance ceiling without
model-load time:

| Backend / beam | Median inference | RTFx | Notes |
| --- | ---: | ---: | --- |
| Native Transformers/PyTorch FP16 / 1 | `188ms` | `53.17x` | Cached model, CUDA |
| Native Transformers/PyTorch FP16 / 2 | `226ms` | `44.30x` | Cached model, CUDA |
| Native Transformers/PyTorch FP16 / 5 | `295ms` | `33.86x` | Cached model, CUDA |
| faster-whisper CUDA FP16 / 1 | `349ms` | `28.67x` | Cached model, CUDA |
| faster-whisper CUDA FP16 / 2 | `371ms` | `26.97x` | Cached model, CUDA |
| faster-whisper CUDA FP16 / 5 | `413ms` | `24.23x` | Cached model, CUDA |
| r5 WebGPU FP16 / 1 | `~0.82-0.83s` | `~12.1-12.2x` | 20 steps, zero GPU downloads |

The native measurements are decode-only after model load; WebGPU includes the
runtime's transcription stages but not the separately reported session load.
No local whisper.cpp executable was available, so no download or fabricated
comparison was added.
