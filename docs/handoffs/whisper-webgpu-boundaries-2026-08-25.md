# Whisper WebGPU Boundary Continuation

Date: 2026-08-25
Workspace: `N:\github\asrjs\speech-recognition`
Browser harness: `N:\github\asrjs\webgpu-agent-test`

## Landed in this continuation

- Split-graph word alignment now follows Whisper/faster-whisper's
  teacher-forced `find_alignment` contract: generated timestamp spans are
  retained for segment context, while the alignment graph receives a
  no-timestamps prompt containing only text tokens.
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
  safely, and aligns only text rows against the attention graph's frame axis.
- Merged-decoder alignment also crops padded frames to the actual audio duration;
  it no longer halves the encoder hidden-state length as a proxy for attention
  frames. Focused regression tests cover the prompt, causal logit rows, cache
  inputs, text-row extraction, and duration crop.

## Validation

Focused unit coverage: 31 tests passed for manifest parsing, beam decode, and
split-graph alignment. Typecheck and build passed.

Real local runner artifacts:

- English `jfk-10s.ogg`, custom `fp16_iofp32` splitgraph model, beam 2:
  Windows OGG conversion, Node inference, and coherent transcript completed.
- Turkish `tr-tdk-18s.wav`, the same model, `language=auto`, beam 2:
  language detection selected `tr` and produced coherent Turkish text.
- English beam 2 with decoder-align word timestamps completed with 16 words.

Independent headless Chrome/WebGPU matrix on the custom
`fp16io-fp16-webgpu` preset (`fp16_iofp32_fp16out` encoder + `fp16` decoder):

| Case                                      |               Total |           RTFx | Steps | KV         | Downloads | Parity                  |
| ----------------------------------------- | ------------------: | -------------: | ----: | ---------- | --------: | ----------------------- |
| EN greedy GPU-KV, 30s                     |           1167.54ms |       25.6131x |    49 | gpu-buffer |         0 | coherent                |
| EN stable beam 2, 30s                     |         10914.095ms |        2.7400x |    98 | cpu        |         0 | oracle                  |
| EN batched beam 2, 30s                    |           8779.44ms |        3.4062x |    49 | cpu        |         0 | exact tokens            |
| TR auto stable beam 2, 18s                |          16593.42ms |        1.1234x |   158 | cpu        |         0 | oracle                  |
| TR auto batched beam 2, 18s               |          14348.08ms |        1.2992x |    79 | cpu        |         0 | exact tokens            |
| EN timestamped stable/batched beam 2, 10s | 4625.065/3946.905ms | 2.1631/2.5347x | 40/20 | cpu        |         0 | exact tokens + 17 words |

The harness reports `check` because it deliberately caps generated tokens;
there were no inference or page errors. GPU-KV greedy retained zero GPU
downloads.

## Timestamp boundary result

The original alignment export had two independent problems. Its manually
unrolled decoder omitted the causal self-attention mask, so teacher-forced
alignment rows could see future text tokens. The runtime also aligned the
generated timestamp spans directly and then clipped the long leading DTW
duration. Together those choices anchored the first word at zero.

The corrected exporter was checked against the regular Transformers decoder:
the selected cross-attention output had maximum absolute difference `0.0`.
With the corrected graph temporarily installed in the browser's actual `fp16`
decoder folder, the 10.004s JFK run produced the same transcript, zero GPU
downloads, and these words:

These measurements are the pre-row-anchor runtime baseline. The graph export
was causal, but the runtime still selected the row after the prompt when it
built the DTW matrix.

| Measurement | Value |
| ----------- | -----: |
| Warm WebGPU total | `779.90ms` |
| RTFx | `12.8277x` |
| First word | `In 2.42–3.00s` |
| Word count | `17` |

The faster-whisper CPU reference begins the same word at about `2.16s`; the
remaining few hundred milliseconds are normal DTW/postprocessing variation,
not the former zero-anchor failure.

The checked-in change is the exporter/runtime fix, not a replacement model
binary. The corrected local graph is at
`N:\models\whisper-align-causal-20260825\decoder_align.onnx`; the checked-out
harness artifacts were restored after validation. The published 4-graph model
must be re-exported with the corrected exporter before this behavior is
available from a remote preset. No model-hosting update was performed.

The complete local export is at
`N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r2`. It contains all
four FP16 graphs, co-located `.onnx.data` files, the causal alignment marker,
and tokenizer/config files. ONNX checker and CPU ONNX Runtime loading passed
for every graph. Installing only its external-data `decoder_align` graph in
the actual browser harness produced the following independent result:

| Measurement | Value |
| ----------- | -----: |
| Warm WebGPU total | `730.51ms` |
| RTFx | `13.695x` |
| First word | `In 2.42–3.00s` |
| Word count | `17` |
| GPU KV / downloads | `gpu-buffer / 0` |
| Warnings | none |

With the restored legacy graph, the same harness emitted both
`whisper.decoder-align-legacy` and
`whisper.decoder-align-legacy-fallback`, and returned generated interpolation
times. This is intentional compatibility behavior until the remote artifact
is re-exported.

The follow-up run after restoring the exact 400-point mel default completed on
the corrected r2 artifact with the same transcript and alignment behavior:

| Measurement | Value |
| ----------- | -----: |
| Warm WebGPU total | `983.37ms` |
| RTFx | `10.1735x` |
| Preprocess | `80.055ms` |
| First word | `In 2.42–3.00s` |
| Word count | `17` |
| GPU KV / downloads | `gpu-buffer / 0` |
| Warnings | none |

### Causal prediction-row correction (2026-08-25)

Whisper decoder row `i` predicts input token `i + 1`. The forced-alignment
sequence used by this package is `[SOT, language, task, ...text, EOS]`, so the
first text token is predicted by the final prompt row (`promptLength - 1`), not
by the first text-token input row (`promptLength`). The split-graph runtime,
merged-decoder runtime, and portable WhisperX-compatible runner now derive and
share this row anchor through `getWhisperForcedAlignmentTextRowStart`.

The same corrected r2 graph was temporarily installed in the actual headless
Chrome/WebGPU harness for an A/B check. The transcript and word count stayed
identical, while the first word moved from the pre-fix `2.42–3.00s` to
`2.10–2.70s`, close to the independent faster-whisper CPU/int8 reference of
`2.16–2.84s`:

| Measurement | Value |
| ----------- | -----: |
| Warm WebGPU total | `775.48ms` |
| RTFx | `12.9008x` |
| Encode / decode | `184.46ms / 470.22ms` |
| Decoder steps | `20` |
| First word | `In 2.10–2.70s` |
| Word count | `17` |
| GPU KV / downloads | `gpu-buffer / 0` |
| Warnings | none |

The corrected browser files were restored after the probe. Focused merged and
split alignment tests pass, including a regression that proves the final
prompt row is used for the first text token. The official Whisper reference
uses the same causal teacher-forced alignment convention in its timing path;
see [Whisper `timing.py`](https://github.com/openai/whisper/blob/main/whisper/timing.py).

## Merged-decoder alignment boundary

The merged-decoder path had a separate, untested alignment implementation that
had drifted from the corrected split-graph contract. It inserted
`<|notimestamps|>` into the teacher-forced sequence, hard-coded a four-token
prompt, read attention rows beginning at row zero, and inferred the frame count
by halving encoder hidden-state positions. The runtime now uses
`[SOT, language, task, ...text, EOS]`, derives the prompt length from that
sequence, reads the causal logit row that predicts each text token, skips prompt
rows when building DTW matrices, and crops the attention frame axis to the
input audio duration.

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

The mel benchmark now reports both contracts explicitly (`n_mels=128`, five
runs on the local Node host): exact 400-point default `177.3ms` for 30s audio
versus experimental 512-point `49.0ms`. The latter remains opt-in because the
speedup changes the model's frequency-bin input contract.

Independent local faster-whisper CPU/int8 on the same 29.904s JFK fixture,
with three warmed measurements per beam, produced:

| Backend / beam | Median inference | RTFx | Text |
| -------------- | ---------------: | ---: | ---- |
| faster-whisper CPU int8 / 1 | `16.014s` | `1.867x` | identical |
| faster-whisper CPU int8 / 2 | `16.055s` | `1.863x` | identical |
| faster-whisper CPU int8 / 5 | `16.259s` | `1.839x` | identical |

Current 10.004s WebGPU beam probes retained exact stable/batched text parity:

| Browser mode | Total | RTFx | Decoder steps | Tokens |
| ------------ | ----: | ---: | ------------: | -----: |
| Stable beam 2 | `6741.01ms` | `1.4841x` | `34` | `18` |
| Batched beam 2 | `6490.19ms` | `1.5415x` | `17` | `18` |
| Stable beam 5 | `12318.94ms` | `0.8121x` | `85` | `18` |
| Batched beam 5 | `10384.57ms` | `0.9634x` | `17` | `18` |

After the immutable-cache sharing change, warmed 29.904s English beam 2
repeats improved while retaining exact stable/batched text parity:

| Browser mode | Total | RTFx | Decoder steps | Change vs prior reference |
| ------------ | ----: | ---: | ------------: | -------------------------: |
| Stable beam 2 | `9571.865ms` | `3.1242x` | `98` | `12.3%` faster |
| Batched beam 2 | `7850.17ms` | `3.8094x` | `49` | `10.6%` faster |

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

| Browser mode | Total | RTFx | Decoder steps | Token parity |
| ------------ | ----: | ---: | ------------: | ------------ |
| Greedy GPU-KV | `1215.745ms` | `24.5975x` | `49` | baseline |
| Stable beam 2 | `8818.83ms` | `3.3910x` | `98` | oracle |
| Batched beam 2 | `7455.44ms` | `4.0111x` | `49` | exact stable |

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
- The local native reference remains faster-whisper CPU/int8 at about `1.84x`
  to `1.87x` RTFx on this fixture. `N:\models\whisper-cpp` contains GGML
  weights but no runnable `whisper-cli`/`main` executable, so no whisper.cpp
  timing is claimed.

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

## Remaining boundary

The local row-anchor boundary is fixed and independently validated, but the
corrected graph must still be regenerated for each published precision variant
and validated on the remote model before the default preset can claim
artifact-level timestamp parity. Merged-decoder end-to-end validation still
needs a timestamped merged graph with `cross_attentions.*` outputs and a broad
English/Turkish reference fixture.

The FireRedASR2S source tree is present at `N:\github\ysdede\FireRedASR2S`,
but no ASR2 checkpoint, `cmvn.ark`, or converted runtime artifact is available
locally. The cached Qwen material is an unrelated text model, so the package's
FireRed VAD support and Qwen ASR documentation remain artifact-gated rather
than being expanded with an unverified runtime.

Keep `experimentalBatchedBeam` opt-in, stable CPU-KV beam as the correctness
oracle, and GPU-KV greedy-only until broader model/reference coverage closes
these boundaries.
