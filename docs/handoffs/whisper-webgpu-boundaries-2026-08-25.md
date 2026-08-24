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
- The GPU-KV policy guard runs before mel preprocessing and encoder inference.
  GPU-KV is still greedy argmax only; beam, `best_of`, and temperature remain
  rejected until cache cloning/reordering is proven correct.

## Validation

Focused unit coverage: 28 tests passed for split-graph alignment and word
duration handling. Typecheck and build passed.

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

## Current performance reference

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

The 30s greedy GPU-KV repeat reached `1080.68ms` / `27.6718x`, with 49
decoder steps and zero GPU downloads. The earlier 30s beam matrix remains the
long-audio parity reference above; one repeat of stable beam 2 did not produce
a result within 166s and was excluded rather than treated as a performance
claim.

## Remaining boundary

The corrected graph must still be regenerated for each published precision
variant and validated on the remote model before the default preset can claim
artifact-level timestamp parity. Merged-decoder timestamp behavior and a
broader English/Turkish reference fixture remain open.

Keep `experimentalBatchedBeam` opt-in, stable CPU-KV beam as the correctness
oracle, and GPU-KV greedy-only until broader model/reference coverage closes
these boundaries.
