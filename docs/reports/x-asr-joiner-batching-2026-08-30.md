# X-ASR streaming: speculative batched joiner decode (2026-08-30)

## Change

`OrtXAsrExecutor.decodeFeatures` previously scored the joiner once per encoder
frame (one tiny graph run per row). The exported joiner graph declares its
leading dimension as `N` (`encoder_out [N,512]`, `decoder_out [N,512]`,
`logit [N,5000]`), so greedy RNNT decode of a chunk's frames against an
unchanged decoder state is row-parallel. The executor now speculatively scores
all remaining frames of the chunk in one batched joiner run: blank rows never
change the decoder state, so the first non-blank row is exactly what the
sequential loop would emit; after an emission the decoder advances and the
frame suffix is re-batched. The result is token-for-token identical to the
previous sequential path by construction.

Safety properties (unit-tested in `tests/x-asr-joiner-batching.test.ts`):

- A joiner run that rejects batched shapes throws and permanently latches
  batching off (`joinerBatchAllowed`); execution continues on the identical
  sequential path in the same call.
- A batch output that is not row-parallel (leading dim or row width mismatch)
  likewise latches batching off and the current frames fall through to the
  sequential path; nothing is consumed from the suspect batch.
- `PipelineAbortedError` from any session propagates; the abort-without-
  corruption contract from `tests/x-asr-decode-abort.test.ts` still passes.
- Row-local token ids are derived by subtracting the row offset from the
  shared absolute-index `argmax`.

## Measurement (real artifact, Chrome headless WebGPU, NVIDIA Blackwell)

Harness: `N:/github/asrjs/webgpu-agent-test/scripts/run-xasr-webgpu.mjs
--streaming --runs=3 --warmup=1 --oracle=exact` on
`/gigaam-audio/jfk-short.wav` (11.29 s, 55 streaming chunks).

| state | median transcribeMs | median RTFx | oracle |
| --- | --- | --- | --- |
| before (sequential per-frame joiner, 2026-08-29) | 8981.1 | 1.2248 | exact pass |
| after, profiled run (--profile) | 7715.5 | 1.4257 | exact pass |
| after, clean 3-run session | 8336.0 | 1.3196 | exact pass |

Joiner dispatches per pass dropped from 311 (=frame rows) to ~88 (one batch
per chunk plus one re-batch per emitted token): `joinerRows=622` across the
two profiled passes. Transcript is byte-identical to the oracle in every run.

The end-to-end gain is modest because the joiner was only ~0.3 s/pass of the
total. The measured profile (per-session totals for 2 passes):

- encoder: 136 runs, 14,983 ms total, ~110 ms/run -> ~90% of transcribe time
- decoder: 188 runs, 757 ms total (~4 ms/run)
- joiner: 175 runs, 626 ms total (~3.6 ms/run)

Result files:

- before: git history of `x-asr-zh-en-160ms-jfk-short-webgpu-stream-chrome.json`
  (recorded 2026-08-29, also in `docs/reports/x-asr-webgpu-streaming-parity-2026-08-29.json`)
- after (clean): `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-stream-chrome.json`
- after (profiled): `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-stream-profiled-chrome.json`

## Validation

- `tests/x-asr-joiner-batching.test.ts`: batched vs forced-sequential parity,
  batch-count reduction, reject-fallback, bad-shape fallback, full tensor
  disposal (4 tests).
- Existing contracts green: abort-safety (3), family/mapping (6), real-artifact
  Node WASM + public streaming (3, `XASR_ONNX_SMOKE=1`).
- Full suite: 1037 passed / 18 artifact-gated skipped; typecheck+build clean.

## Next bottleneck (measured, not guessed)

X-ASR streaming is now encoder-dominated: 68 stateful Zipformer2 runs at
~110 ms each. Candidate follow-ups in priority order: (1) attribute the
per-run encoder cost (dispatch vs 116-tensor GPU state I/O vs kernel time)
with an encoder-only component profile; (2) larger chunk windows (29->40+
frame spans) to cut run count, gated by parity against the 160 ms contract;
(3) fp16 joiner/decoder input path is unnecessary now that dispatch counts
are low. Do not re-port mel/fbank: the 2026-08-29 incremental frontend slice
already removed the quadratic feature cost.

