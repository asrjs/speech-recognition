# Whisper beam-2 GPU-KV decode (2026-08-30)

## Change

Stable beam search previously rejected GPU-KV entirely and ran on the
snapshot-based CPU-KV path: every beam step copied the whole KV to CPU
snapshots (copyAndReleaseWhisperPresentKv) and re-materialized tensors
(cloneDecoderKvDataForInput) - ~48 ms/step x 98 executions.

New opt-in flag `experimentalGpuKvBeam` (a modifier of
`experimentalGpuKvCache`) keeps the KV tensors GPU-resident for beam
search:

- runInit hands the strategy the init present tensors (renamed past.*) with
  no copy/release; runStep feeds them read-only via the prepared-past path
  of runDecoderStepSplit and returns the new gpu present tensors.
- The core beam contract already passes caches opaquely and shares parent
  caches across sibling beams read-only, so no core.ts changes were needed.
- Lifetime: every gpu KV tensor is registered in a generation tracker;
  a tensor not fed/produced for numBeams+1 decoder-step generations is
  provably garbage (superseded by expansion) and is disposed; everything
  is disposed in a finally when the decode ends (success or abort).
- Greedy (numBeams=1) and the CPU-KV stable beam are untouched; batching
  (experimentalBatchedBeam) is disabled while the GPU beam flag is active.

## Validation (Chrome headless, ORT Web 1.29, Blackwell, jfk-30s, fp16io)

- Parity: the GPU-KV beam-2 transcript is byte-identical to the CPU-KV
  stable beam-2 control measured back-to-back today (50 tokens, both
  capped at maxNewTokens=50).
- Performance: GPU beam 1585-1938 ms across five runs (best 1584.7 ms,
  ~18.9x RTFx; median ~1687 ms, ~17.7x) vs CPU beam control 5417.8 ms
  (5.52x) - a 3.2-3.4x end-to-end speedup for beam-2 quality.
- Disposal: three consecutive soak pages completed without gpu-buffer
  double-free or OOM errors. Abort propagates PipelineAbortedError through
  the same finally-disposal block (code path shared with success).
- Both legs measured back-to-back in the same environment state, so the
  relative speedup holds despite the documented ~2x cross-session drift.

## Configuration

- Library: source option `experimentalGpuKvBeam: true` together with
  `experimentalGpuKvCache: true` and a WebGPU decoder backend.
- Harness: `?gpuKv=1&gpuKvBeam=1&numBeams=2`; matrix case
  `en-stable-beam-2-gpu-kv` added to scripts/run-webgpu-matrix.mjs.
- Gating: temperature sampling and best_of remain rejected on GPU-KV paths;
  experimentalBatchedBeam is ignored while the GPU beam flag is active.

## Evidence

- tools/data/results/whisper/beam2-gpu-kv-jfk-30s.json (1584.7 ms)
- tools/data/results/whisper/beam2-gpu-kv-jfk-30s-soak.json (16766 ms page
  total incl. model load; measured transcribe 1676.6 ms)
- tools/data/results/whisper/beam2-gpu-kv-jfk-30s-first-pass.json (the
  first pass hit the encoder .data read documented below and errored)
- tools/data/results/whisper/beam2-cpu-jfk-30s.json (CPU control 5417.8 ms)
- Harness debug note: reading a gpu-buffer tensor's .data getter throws
  even for an instanceof probe - always branch on tensor.location (the
  existing isGpuBufferTensor helper) before any .data access.

## Known boundaries

- The first validation pass errored on encoderHiddenStates.data (gpu
  encoder output) - fixed by the dims-sized placeholder; kept as
  beam2-gpu-kv-jfk-30s-first-pass.json for the record.
- Both beam variants truncate at maxNewTokens=50 on this fixture; the
  transcript is mid-sentence at the cap by design of the probe.
- Batched GPU beam (batch>1 per step) remains future work; scalar GPU beam
  steps already remove the dominant CPU-KV round-trip cost.

## Wider-beam coverage + guard-divergence correction (later the same day)

- Audit finding: the `numBeams <= 2` guard added by the beam-5 failure
  guard (commit 4d55b30) was never actually removed, even though the
  follow-up fix commit (687f39d, 3x prune lag) and docs claimed "all beam
  sizes validated". Every beam-3/beam-5 page hit the greedy-only assert
  and errored; the committed "beam5-gpu-kv-jfk-30s.json" recorded
  decoding.numBeams=2. Lesson: verify evidence JSON fields against the
  claimed config before writing "validated".
- The guard is now genuinely removed (executor.ts gpuKvBeamActive gate).
- Re-measured on the current build (Chrome headless, fp16io, 3x-lag code):
  - beam-3 jfk-30s: 2234.1 ms measurement run (~13.4x RTFx), 50 tokens,
    byte-identical to beam-2 GPU-KV (beam3-gpu-kv-jfk-30s.json)
  - beam-5 jfk-30s: 3090.0 ms measurement run (~9.7x RTFx), 50 tokens,
    byte-identical; this is the first genuine numBeams=5 GPU-KV pass
    (beam5-gpu-kv-jfk-30s.json, overwritten with the real beam-5 run)
  - beam-2 + word timestamps jfk-10s (noTimestamps=0): pass, 16 words with
    sane spans ("In" 0-0.2 s ... "role..." 8.062-9.173 s), same 50-token
    prefix (beam2-gpu-kv-timestamps-jfk-10s.json)
- No premature-disposal errors appeared at beam-3 or beam-5 with the 3x
  lag, confirming the root-cause fix across beam sizes.
