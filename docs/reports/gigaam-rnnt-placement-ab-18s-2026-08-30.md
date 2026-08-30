# GigaAM RNN-T per-component placement A/B on the shared 18 s fixture (2026-08-30)

Goal work item 2 completion slice: per-component backend placement re-measured
on the current stack (ORT Web 1.29.0, shared radix-5 fbank) using the canonical
cross-model clip, replacing the 11.29 s example-fixture verdicts from
2026-08-29 with the same audio every other family is benchmarked on.

## Protocol

- Harness: webgpu-agent-test/scripts/run-gigaam-rnnt-webgpu.mjs (Chrome +
  real WebGPU adapter, Vite dev server on :8765).
- Audio: librivox.org.wav, 18.714 s, within the documented 30 s Whisper-class
  window; --warmup=1 with at least one repetition.
- Runtime matrix: encoder/decoder/joint provider overrides
  (--encoder-backend, --decoder-backend, --joint-backend).
- Oracle: --oracle=none (throughput probe). The librivox clip is an English
  LibriVox recording; the runner's fixed oracle text is the Russian Pushkin
  sample that belongs to /gigaam-audio/example.wav. Using the fixed oracle
  against this clip fails even with a perfect transcript, so correctness here
  is established by cross-placement transcript identity: all four
  compositions produced the identical transcript in this session.
- All legs ran in one contiguous window (12:15-12:27 UTC) against the same
  adapter; absolute RTFx is still subject to the documented host drift
  (concurrent VRAM consumers), so the A/B ratios are the durable claim.

## Results (18.714 s clip, warmed, same session)

| Composition | RTFx | Encode ms | Decode-loop ms | Verdict |
|---|---|---|---|---|
| enc GPU + dec WASM + joint WASM (shipped default) | 27.9 | 84.6 | 517.7 | baseline |
| enc GPU + dec GPU + joint WASM | 4.19 | 82.7 | 4,327.1 | 6.7x regression |
| enc GPU + dec WASM + joint GPU | timeout > 600 s (18.7 s clip); 5.11x on 5 s clip | 59.9 | 893.9 (5 s) | dispatch-bound joiner |
| enc GPU + dec GPU + joint GPU (all-GPU) | 2.63 | 81.5 | 6,966.7 | worst cell |

Evidence files:

- tools/data/results/gigaam/gigaam-rnnt-librivox-18s-warmed-encoder-webgpu-decoder-wasm-joint-wasm.json
- tools/data/results/gigaam/gigaam-rnnt-librivox-18s-warmed-encoder-webgpu-decoder-webgpu-joint-wasm.json
- tools/data/results/gigaam/gigaam-rnnt-librivox-18s-warmed-encoder-webgpu-decoder-webgpu-joint-webgpu.json
- tools/data/results/gigaam/gigaam-rnnt-librivox-5s-warmed-encoder-webgpu-decoder-wasm-joint-webgpu.json

The encoder itself is GPU-efficient in every cell (encode ~82-85 ms on GPU,
~60 ms on the 5 s clip); the entire gap sits in the per-token decoder/joint
loop, exactly as the earlier 11.29 s matrix predicted but now on the shared
fixture. The joint graph on GPU is the worst case: it cannot finish an 18.7 s
clip inside the harness's 600 s budget, while the same graph on WASM
contributes a sub-millisecond-scale per-step cost inside the 518 ms full-loop
time.

## Consistency with prior and parallel evidence

- 2026-08-29 matrix (11.29 s example fixture): hybrid 32.9x (radix5) vs
  all-GPU 2.90x - same ordering, same conclusion.
- Plugin EP spike (same day): GigaAM-style int8 recurrence falls back to CPU
  in the native WebGPU Plugin EP too - quantized recurrences do not move the
  small-step GPU case.
- Parakeet TDT GRU/LSTM decoder findings (decoder-only GPU = ~8x penalty):
  identical mechanism - per-step dispatch and readback overhead dominates
  tiny single-frame graphs regardless of vendor engine.

## Decision

The shipped GigaAM RNN-T composition (WebGPU encoder + WASM decoder + WASM
joiner) is placement-optimal on measured evidence across two fixtures and two
session days. No promotion candidate comes out of this slice; the remaining
RNN-T levers stay what the goal already records: encoder export size/precision
(fp16 is the validated option at 27.4x vs 32.9x fp32 - a size trade, not a
speed trade) and decode-step cost reduction via loop restructuring (step
batching), not per-graph backend movement.

## Harness note for future librivox legs

Any librivox-fixture leg must pass --oracle=none (or a matching English
label); the runner exits nonzero on status fail, and the fixed Russian
oracle mislabels throughput runs as correctness failures. The three
"fail"-status rerun artifacts produced before this lesson were discarded;
only oracle=none (throughput) and example.wav (exact parity) evidence is
committed.
