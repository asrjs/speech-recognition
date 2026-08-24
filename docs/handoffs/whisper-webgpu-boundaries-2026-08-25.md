# Whisper WebGPU Boundary Continuation

Date: 2026-08-25
Workspace: `N:\github\asrjs\speech-recognition`
Browser harness: `N:\github\asrjs\webgpu-agent-test`

## Landed in this continuation

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

Focused unit coverage: 28 tests passed across beam decode, splitgraph decode,
GPU-KV policy, and runner CLI/KV-shape tests. Typecheck and build passed.

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

## Remaining boundary

The local faster-whisper CPU reference produced the same English words but
different timestamp anchoring on `jfk-10s.ogg` (reference first word about
2.39s; this decoder-align/interpolation path starts at 0s). This is a real
alignment-quality difference, not a reason to change decoder tokens or the
GPU-KV policy. Compare the decoder-align export and optional Wav2Vec2 refine
path against the reference before changing timestamp semantics.

Keep `experimentalBatchedBeam` opt-in, stable CPU-KV beam as the correctness
oracle, and GPU-KV greedy-only until broader model/reference coverage closes
these boundaries.
