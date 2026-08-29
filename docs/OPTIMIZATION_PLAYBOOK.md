# ASR optimization playbook (cross-family)

Distilled from the Whisper Large V3 2x to 26x case study, the Parakeet TDT
hybrid-placement work, and the 2026-08-29/30 family matrices. Apply it to
every model after correctness is verified. Correctness first, optimization
second, measurement throughout.

## 1. Measurement discipline

1. Attribute before editing. Measure the stage (preprocess / encode /
   decode), then the algorithm inside the dominant stage. A constant-factor
   pass on the wrong stage measures as zero (the GigaAM windowing loops
   moved nothing because the FFT consumed 92% of preprocessing).
2. Same-launch pairing for A/B claims. On the benchmark host, decode-phase
   timing drifts up to ~20% and total medians ~11% across browser sessions.
   Run control and candidate in the same launch (or use the Node
   microbenchmark harness for JS hot paths); never compare cross-session
   phase numbers.
3. Warm before measuring. At least one same-session warm-up run, then
   report the median of 3-5 measured runs; report warm-up separately.
4. Preserve oracles. Fixed-transcript oracles (exact or normalized) gate
   every optimization. Throughput probes on synthetic audio use
   oracle=none and are labeled as such.
5. Keep reproducible microbenchmarks. tools/scripts/benchmark-*.mjs are the
   authoritative evidence for JS hot paths; browser JSONs under
   tools/data/results/ are the authoritative evidence for end-to-end
   claims. Commit both with the change.

## 2. Placement rules (workload-specific, measured)

Placement is not a global preference; it follows the graph shape:

- One-frame recurrent decode loops (GRU/TDT predictor, RNNT decoder+joint):
  WASM. WebGPU per-step dispatch loses by 10-16x (Parakeet v3 WASM ~36x vs
  full-WebGPU 5-6x; GigaAM RNN-T decode 236 ms WASM vs 3749 ms WebGPU).
- Single encoder graphs: WebGPU. SenseVoice encode is ~10x faster than
  8-thread WASM (205 vs 1945 ms); GigaAM CTC runs ~61x on WebGPU.
- GPU-resident recurrent state (decoderStateOutputLocation='gpu-buffer')
  helps WebGPU decoder loops (12-21% end-to-end on two browsers, soak
  verified) but is irrelevant to WASM decoders. Promotion still awaits a
  non-NVIDIA adapter.
- Preprocessor and WASM decode threading: enable cpuThreads but measure;
  threading wins appeared only in some sessions and stayed inside variance
  for one-frame steps.

## 3. Precision and quantization

- Quantization is a size/memory decision first; speed must be benchmarked,
  never assumed. v2 int8 was slower than fp32 in some sessions.
- Proven wins: fp16 encoders on WebGPU; int8 decoders on WASM for
  vocabulary-projection-dominated loops (Parakeet v3 fp32 22.8x to int8
  37.2x, exact parity).
- Vocabulary projection dominates TDT/RNNT step cost: v3 projects to 8198
  classes vs v2's 1030, and that alone explains the 2.4x per-step gap.
- Browser fp32 graphs with external-data files are blocked by ORT Web
  mounting (Module.MountedFiles is not available); ship single-file fp16
  or int8 variants for browser use.

## 4. JavaScript hot-path discipline

- Borrow typed-array views instead of copying when ownership is bounded by
  the consuming step (Parakeet logits borrow, ~17%).
- Gate optional work: confidence/entropy traversals behind returnConfidence
  (no end-to-end delta for Parakeet because views were already borrowed -
  measure before claiming).
- Remove ?? coalescing from in-bounds tight loops via hoisted exact-length
  fast paths with guarded slow paths (CTC argmax 1.29x, bit-identical).
- Verify optimizations bit-identically where semantics allow (checksums,
  printed digits, naive-oracle tests).

## 5. Algorithmic fixes beat constant factors

- Check the algorithm class first: Bluestein chirp-z spent three 1024-point
  FFTs per 320-point transform; a direct radix-5 x power-of-two
  decomposition (RadixFivePowerOfTwoFft) cut preprocessing 2.84x and
  GigaAM RNN-T end-to-end 449 to 343 ms. Any nFft that is not a power of
  two deserves this check.
- Prefer precomputed tables over per-frame trig (the first radix-5 version
  was slow despite correct structure because cos/sin ran per frame; a
  second sign bug produced wrong output - verify against a naive DFT
  oracle before claiming any FFT change).

## 6. WebGPU execution lessons (from Whisper 2x to 26x and Qwen probes)

- Keep decoder state and logits GPU-resident; scalar-only readback. The
  Whisper case study's decisive moves were GPU-resident KV, avoiding
  JS-owned typed-array round-trips per step, and bounded top-k instead of
  per-beam full log-softmax.
- Graph capture and dimension overrides are worth a probe but expect
  EP-partitioning rejections on dynamic-dimension graphs (Qwen).
- Graph surgery can backfire: the Qwen ArgMax-output probe cut output
  handling 97% yet lost 61% end-to-end - provider reduction kernels and
  readback interact non-obviously. Record PERFORMANCE_NOT_VIABLE results
  to avoid re-running them.
- Browser entry points matter: keep the plain onnxruntime-web import on the
  all bundle and the /webgpu subpath on the webgpu bundle.

## 7. New-port optimization checklist

1. Baseline the exact artifact/backend on the shared fixtures with warm-up
   and median-of-N; record load/encode/decode/preprocess phases.
2. Compare placement cells: encoder backend, decoder backend, state
   location, threads. Use same-launch pairing.
3. Inspect nFft and any non-power-of-two transforms; check preprocessing
   algorithm class.
4. Profile the decode/post-processing JS hot paths with Node
   microbenchmarks; apply borrowing/gating/hoisting with bit-identical
   verification.
5. Probe precision variants that exist locally (fp16/int8) with parity
   oracles before and after.
6. Record negative results (PERFORMANCE_NOT_VIABLE) in the report with the
   measured numbers so future ports skip them.
7. Update the family matrix in GOAL_PROMPT.md and commit evidence JSONs
   with the change.

## Case study references

- Whisper: docs/Whisper-Optimizations.md, docs/OPTIMIZATION-SPRINT-REPORT.md,
  docs/whisper-splitgraph-local.md
- Parakeet TDT: docs/reports/parakeet-tdt-decoder-quantization-matrix-2026-08-29.md,
  docs/reports/parakeet-tdt-gpu-state-second-browser-2026-08-29.md,
  docs/handoffs/parakeet-tdt-webgpu-ep-2026-08-29.md
- GigaAM: docs/reports/gigaam-rnnt-placement-threads-matrix-2026-08-29.md,
  docs/reports/gigaam-preprocess-radix5-fft-2026-08-30.md
- SenseVoice: docs/reports/sensevoice-placement-correction-2026-08-29.md,
  docs/reports/sensevoice-decode-optimization-2026-08-29.md

