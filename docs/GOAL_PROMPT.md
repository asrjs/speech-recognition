# Codex Goal Prompt: Build a Best-in-Class Web ASR Library

You are continuing development of N:\github\asrjs\speech-recognition,
the @asrjs/speech-recognition TypeScript library.

Your mission is to make @asrjs/speech-recognition a best-in-class web ASR
library: accurate, fast, memory-efficient, reliable, easy to integrate, and
pleasant to use in both browser and local Node.js applications.

Work across the whole product, not only model porting. Advance the areas with
the highest expected user value:

1. a coherent, stable, speech-first TypeScript API;
2. production-quality browser and Node.js runtime behavior;
3. high-value, fully verified ASR model families and presets;
4. WebGPU/WASM performance, memory, loading, caching, and lifecycle;
5. realtime, streaming, long-audio, VAD, chunking, timestamps, alignment,
   language detection, confidence, and quality behavior;
6. excellent examples, demos, documentation, benchmarks, and diagnostics;
7. reusable model-porting, conversion, debugging, and parity infrastructure.

This is a single-package, ESM-first, speech-focused, headless and
framework-neutral runtime. It is not a generic model zoo or multimodal
framework.

## Completed (2026-08-30 recent slices)

The following work is done and pushed on `main` (backup branch
`backup/pre-browser-webgpu-state-push-2026-08-29`). Details:
`docs/handoffs/asr-four-families-2026-08-27.md`.

Browser WebGPU state slice (`4856dc8`, `f965e6e`):

- X-ASR GPU-resident streaming state: 32.10s → 9.26s (71.14% reduction /
  3.46× speedup)
- Qwen GPU KV validation: median 7.12s → 4.86s (31.75% reduction / 1.47×
  speedup)
- Exact transcript parity preserved in both A/B comparisons
- Fixed Qwen Node-hosted ORT Web WASM external-data loading; real 1.50 GB
  fp16 decoder passes again
- Browser bundles no longer attempt to include native ORT `.node` binaries
- X-ASR WASM remains faster at 6.84s for small streaming graph — backend
  selection remains workload-specific
- Validation: 1003 tests passed, 18 artifact-gated skips; Qwen real artifacts
  WASM+Node WebGPU passed; X-ASR real artifacts WASM/WebGPU/stateful
  streaming passed; build/typecheck clean; lint 0 errors 11 warnings;
  browser root import graph and Chrome/WebGPU compatibility passed
- ORT 1.29.0 note: npm reported `ETARGET` at the time of this slice; the
  upgrade was completed separately in the slice below

ORT Web 1.29.0 stable upgrade (2026-08-29):

- `onnxruntime-web` 1.27.0-dev → 1.29.0 stable with `onnxruntime-common`
  1.29.0; the `onnxruntime-node` nightly stays pinned (nested 1.24-dev
  common). npm's CLI cannot reach the registry on this host, so the install
  used the registry tarballs directly with hand-authored lock entries
  carrying the real resolved/integrity values
- Validation: full suite 1008 passed / 18 artifact-gated skips; typecheck
  clean; lint 0 errors / 11 warnings; production build clean
- Lockfile consistency follow-up (2026-08-29): the root lock declaration now
  pins `onnxruntime-web` to the same exact `1.29.0` version as `package.json`
  and records the existing Node `>=22` engine. Offline `npm install
  --package-lock-only` and `npm ci --dry-run` both complete successfully;
  `onnxruntime-node` retains its intentionally separate nightly
  `onnxruntime-common` dependency.
- Real-artifact backend suites all green on 1.29.0: GigaAM CTC 4/4, GigaAM
  RNN-T 2/2, SenseVoice 3/3, X-ASR 3/3 including public stateful streaming,
  Qwen 2/2 including the 1.5 GB fp16 decoder external-data mounts
- Chrome WebGPU matrix, exact transcript parity on every family:
  Qwen GPU-KV 3-run median 4,558.9 ms / RTFx 2.41 (about 6% faster than the
  nightly's 4,856 ms / 2.27); GigaAM CTC fp16 281 ms (RTFx 39.8);
  SenseVoice 701 ms (RTFx 15.7); GigaAM RNN-T 6,163 ms (RTF 0.55); X-ASR
  10,103 ms single-run (WASM remains faster for that graph; placement
  verdicts stay workload-specific); Qwen sequential WASM fp16 browser leg
  passed
- Evidence: fresh harness JSONs plus `tools/data/results/qwen/`,
  `tools/data/results/x-asr/`, `tools/data/results/gigaam/`,
  `tools/data/results/sensevoice/`

Qwen WebGPU entry-point boundary refresh (2026-08-29):

- A reproducible Chrome/ORT 1.29 failure (corrupted 13-token output with
  successful session creation) was traced to the sibling harness aliasing both
  `onnxruntime-web` and `onnxruntime-web/webgpu` to `ort.all.bundle.min.mjs`.
  The corrected harness keeps the plain import on the all bundle and maps the
  WebGPU subpath to `ort.webgpu.min.mjs`.
- With the corrected entry point, the official dynamic encoder plus explicit-KV
  decoder restored exact 30-token parity. Three same-session GPU-KV runs had a
  1,854.6 ms median (`5.93x` RTFx); CPU-KV controls had a 3,880.65 ms median
  (`2.83x` RTFx), for a measured `2.09x` GPU-KV speedup on NVIDIA Blackwell.
- This is a browser runtime-entry/harness correction, not a model or graph
  algorithm change. Keep alias separation as a browser acceptance invariant;
  repeat on another browser/adapter before changing a public preset.
  Evidence: `docs/reports/qwen3-asr-webgpu-bundle-boundary-2026-08-29.json`.

Qwen model-specific decoder phase profile (2026-08-29):

- Added phase telemetry to the official stacked and legacy decoder loops:
  input/feed construction, ORT `session.run()`, output/logit handling, step
  count, and cache location. The telemetry is part of the canonical transcript
  metrics and does not alter tensor ownership or output semantics.
- Real Chrome/WebGPU controls (ORT Web 1.29.0, NVIDIA Blackwell, exact
  official artifacts) show a warmed GPU-KV median of `1753.825 ms` / `6.2722x`
  RTFx for the 11-second fixture, with 30/30 exact tokens. Decoder-step work is
  dominated by ORT execution: `1520.570 ms` in `session.run()` versus
  `0.235 ms` feed construction and `24.240 ms` output handling.
- The CPU-KV control is also exact at `3885.960 ms` / `2.8308x` RTFx; GPU-KV
  is a measured `2.2157x` total and `2.3604x` step-loop win. Reusing mutable
  one-token input tensors was rejected because feed construction is only
  `0.0152%` of step time and the ownership risk is not justified.
- The next Qwen target is decoder-step graph/WebGPU EP execution (fusion,
  graph capture, dispatch, and kernel behavior), not JavaScript allocation.
  Preserve exact-token, cache-lifecycle, and browser entry-point controls.
  Evidence: `docs/reports/qwen3-asr-webgpu-decoder-profile-2026-08-29.json`.

Qwen decoder graph-capture compatibility probe (2026-08-29):

- Added an opt-in `decoderGraphCapture` source flag plus optional
  `decoderFreeDimensionOverrides`, with the same narrow retry/fallback
  boundary used by Whisper. Capture is never enabled by default.
- Real Chrome/WebGPU ORT 1.29.0 rejected capture for both official stacked
  decoder sessions because not all nodes partitioned to
  `WebGpuExecutionProvider`; the fallback preserved exact 30-token parity.
  Cold load was `62.52 s` with capture requested versus `35.74 s` regular, so
  this is a measured compatibility rejection, not a speedup.
- The decoder has dynamic `past_len`/`present_len` dimensions. Revisit only
  with a static-shape export or a future EP partitioning change, and compare
  full-cache memory traffic before promotion.
  Evidence: `docs/reports/qwen3-asr-webgpu-graph-capture-2026-08-29.json`.

Qwen decoder ArgMax graph-surgery probe (2026-08-29):

- Added the reusable `append_argmax_output.py` tool and optional executor
  support for an INT64/INT32 `next_token_id` output. The candidate removes
  the full 151,936-wide logits graph output while preserving the original
  external-data shards; legacy graphs still use the validated logits fallback.
- Native ONNX Runtime CPU session creation accepted both candidate graphs, and
  the real Chrome/WebGPU run preserved exact 30-token JFK parity. Five-run
  warmed controls measured 1,623.058 ms / 6.7778x RTFx for the official graph
  versus 2,619.933 ms / 4.1992x for ArgMax-only. Output handling fell 97.05%
  (22.225 -> 0.655 ms), but decoder `session.run()` rose 67.08% and total
  transcription rose 61.42%; classify this as `PERFORMANCE_NOT_VIABLE` and do
  not promote it. The result demonstrates that a smaller readback can lose to
  a provider reduction kernel, and should guide future graph/EP experiments.
  Evidence: `docs/reports/qwen3-asr-webgpu-argmax-surgery-2026-08-29.json`.

Parakeet TDT WebGPU EP probe and decode hot path (2026-08-29):

- Corrected Chrome A/B confirms the Parakeet v3 GRU decoder graph runs on both
  WebGPU and WASM with ORT Web 1.27 nightly and 1.29.0 stable; the raw TDT
  duration head is excluded from vocabulary argmax selection
- Opt-in GPU-state outputs reproduce the token sequence and reduce the
  decoder-only warm-step microbenchmark by roughly 29–41%, but disposal
  surfaced an ORT `null function` failure, so production placement remains
  unchanged until lifecycle and full-model parity are proven
- TDT now borrows float32 logits views, uses the confidence-only softmax path,
  and removes redundant disposal promise wrappers; the 5,000-run Node harness
  records the before/after controls in
  `docs/handoffs/parakeet-tdt-webgpu-ep-2026-08-29.md`
- NeMo TDT now accepts `returnConfidence` (default true). Setting it false
  skips the per-step full-vocabulary softmax/entropy pass after token and
  duration argmax, leaving the native confidence summary undefined. This is a
  backward-compatible throughput-only path that matches the `wav2vec2` option
  and the gated softmax in the faster `parakeet.js` reference.
- Validation after this slice: full suite 1008 passed / 18 artifact-gated
  skips; typecheck and build clean; lint remains 0 errors / 11 warnings
- Corrected full-model browser probe: native WASM controls for v3 fp16/fp16,
  fp16/fp32, and int8/fp32 all reproduce the exact 91-token transcript, while
  browser target-rate runs reach 6.98–17.47x RTFx but emit only 74
  tokens. A follow-up with deterministic `native-rate` linear WAV resampling
  reaches exact 91-token parity at 18.66x RTFx (fp16/WebGPU encoder, ONNX
  preprocessor, WASM/fp32 decoder); the same audio strategy makes the full
  WebGPU decoder exact at a three-run median 5.64x RTFx and int8/WebGPU
  encoder + int8/WASM decoder exact at 2.02x RTFx. Treat target-rate numbers
  as a measured preprocessing mismatch; treat int8 as a memory/size option
  rather than an assumed speedup. These are not blanket quantization or
  placement wins. The earlier
  “more than ten minutes” fp16 session-create note was invalid because the old
  page failed before valid model creation; the corrected harness reaches
  model-ready in 7–12 seconds. The fp32 browser control remains blocked by ORT
  Web external-data mounting (`Module.MountedFiles is not available`). Full
  details and JSON controls are in
  `docs/reports/parakeet-tdt-v3-browser-full-model-2026-08-29.md`.
- The reusable Parakeet Chrome runner now defaults to deterministic
  `native-rate` linear WAV preparation; `--audio-strategy=target` is reserved
  for an explicit resampler diagnostic. This keeps future browser acceptance
  runs on the same audio contract as the native reference by default.
- Its `--repeat=N` mode now records same-session warm-up, parity, and
  `performance.memory` snapshots. Eight-run probes stayed exact: hybrid
  fp16/WebGPU-encoder + WASM-decoder warmed to about 25x RTFx, full WebGPU
  decoding to about 6x; GC caused a large heap drop, so longer soak/disposal
  checks remain required before lifecycle promotion.

Lifecycle hardening and cancellation slice (`8552eec`, `e8624e6`):

Parakeet TDT decoder quantization and v2/v3 matrix (2026-08-29):

- Reproduced the historical parakeet.js throughput band in the current
  browser harness: on the shared 18.714 s LibriVox fixture, v2 hybrid runs
  reach ~35-41x RTFx, confirming the 45-90x memories were v2 + longer clips,
  not a library regression. Session-to-session variance on this host is
  ~10-15%.
- INT8 decoder is the dominant, reproducible v3 win: fp16/WebGPU encoder +
  int8/WASM decoder measures 508 ms median / 37.2x RTFx with the exact
  91-token transcript preserved, versus 832 ms / 22.8x for the fp32 decoder
  probe; decode phase roughly halves (565 -> 256 ms). The library browser
  default (int8 decoder) is validated as the fast exact path; the slow
  ~18-28x observations came from parity probes pinning fp32.
- The v3 GRU decoder graph is genuinely heavier per step than v2 at fp32
  (~4.3-5.2 ms vs ~2.1-2.9 ms); INT8 recovers most of the gap.
- The returnConfidence=false gate is a measured end-to-end no-op (within
  session noise); it stays an opt-in throughput option with no speed claim.
- The fp32 browser encoder control remains blocked by ORT Web
  external-data mounting (Module.MountedFiles is not available) on both
  v2 and v3.
- Evidence: docs/reports/parakeet-tdt-decoder-quantization-matrix-2026-08-29.md
  and tools/data/results/nemo-tdt/parakeet-tdt-v*-librivox-18s.json

Parakeet TDT clip-length scaling and v2/v3 decoder gap (2026-08-29):

- A 38.04 s synthetic throughput fixture (LibriVox clip doubled with a
  0.6 s silence gap, oracle disabled) shows RTFx does not grow with clip
  length in this range: v3 int8 hybrid measures 35.8x on 38 s versus 37.2x
  on 18.7 s. Encode and decode both scale linearly; fixed overhead is
  already small.
- v2 reproduces the historical headline band with the current library:
  55.4x RTFx on the 38 s clip (696.3 ms median) with the same hybrid
  composition. There is no library regression behind the old 45-90x
  memories.
- The entire v2/v3 gap is the decoder graph: encoders are identical
  (~280 ms), step counts match (202 vs 203), but per-step decode cost is
  ~1.47 ms (v2) vs ~3.48 ms (v3) - a 2.4x heavier step - and v3 emits more
  tokens for the same audio (182 vs 153). The v3 decoder (vocabulary
  projection size, GPU-state placement, step batching) is the next
  model-specific optimization target with the highest measured leverage.
- Structural root cause: the two decoder_joint graphs are otherwise
  identical (2x LSTM(640) prednet, same ops and states); the gap is the
  vocabulary - v2 projects to 1,030 classes, v3 to 8,198 (~8.5M extra
  parameters in embedding + joint MatMul). This is intrinsic to v3's
  8192-class SentencePiece vocabulary, not a graph defect. int8 WASM
  (~36x) is near this graph's practical ceiling here; the remaining
  high-leverage lever is GPU-state placement (29-41% decoder-only win,
  gated on the ORT disposal lifecycle fix and second-adapter soak).

Parakeet TDT GPU-state second-browser and soak gates (2026-08-29):

- The harness runner now supports --browser=edge (reusable second-browser
  acceptance path). Chrome ANGLE vulkan and gl backends are unavailable on
  this host (WEBGPU_NO_ADAPTER), so the second-engine evidence comes from
  Edge on the same NVIDIA Blackwell/D3D11 adapter.
- Edge library-path A/B: GPU-state decoderStateOutputLocation='gpu-buffer'
  measured 2214.5 ms median / 8.47x RTFx versus 2512.5 ms / 7.46x for the
  cpu control - an 11.9% latency reduction reproducing the Chrome result
  (17.3%), with the exact 91-token transcript and zero disposal errors on
  both browsers.
- Edge lifecycle soak: 8 same-session GPU-state runs, 8/8 exact, transcribe
  times stable in a 1860-2164 ms band with no drift, JS heap showing normal
  GC behavior (no monotonic leak signature), and clean model/runtime
  teardown.
- Both stated promotion gates (second browser, longer soak) have now
  passed. The placement remains opt-in solely because all evidence is one
  GPU vendor (NVIDIA Blackwell/D3D11); AMD/Intel and non-Chromium engines
  are untested. Promote after a non-NVIDIA adapter pass or an explicit
  decision to accept single-vendor evidence.
- Evidence: docs/reports/parakeet-tdt-gpu-state-second-browser-2026-08-29.md
  and tools/data/results/nemo-tdt/parakeet-tdt-v3-webgpu-dec-fp32-*-edge-*.json

Parakeet TDT preprocessor A/B and variance methodology (2026-08-29):

- The old parakeet.js short-clip lesson (JS mel beat ONNX preprocessing)
  does not transfer to 18.7 s clips: JS reliably saves ~20 ms at the
  preprocess stage but the end-to-end median is inside cross-session
  noise, with exact parity on both paths. Keep the ONNX preprocessor
  default; revisit JS mel only for short-clip latency products.
- Cross-session variance is now bounded with three same-config sessions:
  medians 508.1 / 636.7 / 574.0 ms (about +-11%) and decode-phase averages
  255.7 / 385.9 / 312.6 ms (about +-20%). Phase-level A/B claims on this
  host require same-launch paired sessions; cross-session decode
  comparisons are unreliable.
- Heap snapshots were nearly identical between preprocessor modes, ruling
  out GC pressure as the decode-phase confound.

GigaAM RNN-T placement and threads matrix (2026-08-29):

- Re-measured with per-component provider overrides on the 11.29 s exact
  Russian oracle fixture: the default hybrid (encoder WebGPU, decoder+joint
  WASM) runs at 449.3 ms median / 25.13x RTFx with exact parity - the
  family is healthy, not the 0.55x of the early all-WebGPU matrix entry.
- The all-WebGPU composition still measures 3889.4 ms / 2.90x: the tiny
  per-token decoder/joint steps pay a 16x decode-loop penalty on GPU
  (3748.8 ms vs 236.0 ms). Same one-frame-step lesson as Parakeet TDT;
  hybrid placement is validated by a measured 8.7x end-to-end gap.
- The GigaAM browser runner no longer hardcodes cpuThreads: 1; the CLI
  exposes --cpu-threads=N. Eight threads left the end-to-end median flat
  (451.8 vs 449.3 ms); the decode drop sits inside documented variance.
- Preprocessing is JS-only (no ONNX preprocessor ships with the model) and
  measured 75-131 ms across sessions (17-29% of total). An ONNX/WebGPU
  fbank export mirroring the Parakeet nemo128 pattern is the remaining
  stage lever; an int8/fp16 encoder export (844 MB fp32 today) is the
  VRAM/size lever. [Update 2026-08-30: the JS fbank now uses the shared
  radix-5/mixed-radix FFT, so the remaining RNN-T levers are the encoder
  export (fp16/int8) and decode step cost, not mel math.]
- Evidence: docs/reports/gigaam-rnnt-placement-threads-matrix-2026-08-29.md
  and tools/data/results/gigaam/v3-rnnt-*.json

SenseVoice placement correction (2026-08-29):

- Re-measured on the 11.3 s jfk-short exact-oracle fixture: the WebGPU
  default runs at 435.0 ms median / 25.29x RTFx, superseding the stale
  701 ms / 15.7x matrix entry. No regression.
- WebGPU placement is decisively correct for this single-graph encoder
  model: encode is roughly 10x faster than 8-thread WASM (205.5 vs
  ~1945 ms). This is the mirror image of the one-frame-step families
  (Parakeet TDT, GigaAM RNN-T) where WASM wins the decode loop - the
  workload-specific placement rule now has measured evidence in both
  directions across four families.
- decodeMs (160.9 ms) is JS post-processing (full-vocabulary argmax, CTC
  collapse with spans, tokenizer decode, confidence/timing construction),
  not an autoregressive loop. It is the next phase-level target and best
  measured with the Node hot-path microbenchmark harness.
- The phase-sum vs native-total gap is ~10 ms, so the logits readback is
  essentially attributed. The model ships fp32-only (894 MB); int8/fp16
  export remains the VRAM/size lever.
- The SenseVoice browser runner now accepts backend and cpuThreads options
  (--backend=wasm, --cpu-threads=N), matching the other family harnesses.
- Evidence: docs/reports/sensevoice-placement-correction-2026-08-29.md
 and tools/data/results/sensevoice/small-{webgpu,wasm-8t}-jfk-3run.json

SenseVoice decode hot-path optimization (2026-08-29):

- argmaxAndSelectedLogProbs (shared by SenseVoice and GigaAM CTC) measured
  70.18 ms p50 at SenseVoice scale (187 x 25055) and now runs 54.26 ms
  (1.29x) after removing the redundant rowMax tracking (bitwise identical
  to bestValue), hoisting Math.exp/Math.log, and using contiguous
  typed-array access on a hoisted exact-length fast path; outputs are
  bit-identical and the full suite stays green (1017 passed / 18 gated
  skips).
- A reproducible microbenchmark lives at
  tools/scripts/benchmark-ctc-decode.mjs; per the same-launch variance
  methodology, the Node microbenchmark - not a noisy browser phase A/B -
  is the authoritative before/after evidence for this hot path.
- Remaining decode cost was attributed to the intrinsic 4.68M Math.exp
  calls of exact log-softmax; an exact underflow-threshold skip
  (delta <= -40) would help raw-logit graphs but not SenseVoice's narrow
  log-prob range. SUPERSEDED 2026-08-30: the fp16 exp lookup-table fast
  path (see the SenseVoice fp16 CTC decode hot-path slice below) removes
  both the fp16->fp32 conversion and the per-element exp calls.

GigaAM shared preprocessing radix-5 FFT (2026-08-30):

- Root-caused GigaAM preprocessing cost: for the non-power-of-two nFft
  (320), MedAsrJsPreprocessor used Bluestein's chirp-z, which spends three
  1024-point FFTs per frame. The FFT alone was 64.22 of 69.49 ms p50 in a
  Node microbenchmark at the real 11.29 s / 64-mel / nFft-320 shape.
- Added RadixFivePowerOfTwoFft (direct Cooley-Tukey for N = 5 * 2^m) with
  fully precomputed twiddle and 5-point factor tables; selection is
  power-of-two, then radix-5, then Bluestein fallback. Preprocess process()
  dropped 69.49 -> 24.51 ms p50 (2.84x) with an identical feature checksum
  and 2.659e-12 max relative agreement vs Bluestein.
- Browser end-to-end with exact oracles: GigaAM RNN-T 449.3 -> 342.8 ms
  (25.1x -> 32.9x), preprocess phase 130.8 -> 31.7 ms; GigaAM CTC 180.3 ms
  / 61.0x exact. The win applies to every family using this shared frontend
  with nFft = 5 * 2^m.
- Lesson recorded: stage attribution must precede hot-path edits - an
  earlier constant-factor pass on the windowing/mel loops moved nothing
  because the FFT dominated; measure the stage, then the algorithm.
- Evidence: docs/reports/gigaam-preprocess-radix5-fft-2026-08-30.md,
 tools/scripts/benchmark-gigaam-preprocess.mjs, and
 tools/data/results/gigaam/*radix5*.json

Cross-family optimization playbook (2026-08-30):

- Consolidated the Whisper 2x to 26x case study, the Parakeet TDT
  hybrid-placement work, and the 2026-08-29/30 family matrices into
  docs/OPTIMIZATION_PLAYBOOK.md: measurement discipline (variance bounds,
  same-launch pairing, stage attribution), placement rules with the
  measured both-direction evidence, precision/quantization rules, JS
  hot-path discipline, algorithmic-fix precedence (radix-5 FFT), WebGPU
  execution lessons, and a new-port optimization checklist.
- SenseVoice preprocessing measured at the same time: 22.33 ms p50 in Node
  for the 11.29 s shape (nFft 512 is already power-of-two fast; LFR/CMVN
  included). No algorithmic target remains there; the earlier browser
  57.6 ms is environment overhead, not a code lever. Recorded to prevent
  re-investigation.
- Evidence: docs/OPTIMIZATION_PLAYBOOK.md and
 tools/scripts/benchmark-sensevoice-preprocess.mjs

GigaAM RNN-T encoder fp16 quantization probe (2026-08-30):

- Produced a local fp16 encoder (442.8 MB vs 844 MB fp32) with
  onnxruntime.transformers.float16 (keep_io_types=True) via the
  reproducible tool tools/scripts/convert_gigaam_rnnt_encoder_fp16.py.
- Numerical parity on real example.wav features (library preprocessor,
  ORT CPU): cosine 0.9999987, max abs diff 2.3e-3.
- Browser end-to-end: exact Russian oracle parity on both encoders; fp16
  load is ~34% faster (8.0 -> 5.3 s) but encode is ~2x slower on WebGPU
  (70.1 -> 130.7 ms), net 28.7x -> 27.4x RTFx.
- Verdict: fp16 is a valid size/load/VRAM alternative, not a speed win;
  fp32 stays the default. The harness runner now accepts --encoder-file
  for alternate-artifact probes.
- Evidence: docs/reports/gigaam-rnnt-encoder-fp16-probe-2026-08-30.md and
  tools/data/results/gigaam/v3-rnnt-fp16-enc-gpu-decwasm-librivox.json

GigaAM RNN-T encoder INT8 probe (2026-08-30):

- Dynamic per-channel QInt8 conversion was tested against the untouched
  844.1 MB fp32 encoder. A MatMul-only graph is 320.1 MB (62.1% smaller);
  the initial MatMul+Conv graph was rejected because ORT CPU has no
  ConvInteger(10) implementation. The converter defaults to the supported
  MatMul-only surface and records the operator choice explicitly.
- Captured official features preserve output shape/length ([1,768,282] /
  282), but encoder values show material quantization drift (max abs
  0.471812, cosine 0.992093). Both exact fixed Russian transcripts still
  pass in the browser, so transcript parity is recorded separately from
  numerical parity.
- Chrome WebGPU, ORT Web 1.29, Blackwell, 11.29 s example.wav, warmup+3:
  fp32 WebGPU hybrid 449.3 ms / 25.13x; MatMul-only INT8 WebGPU
  2676.1 ms / 4.22x. The dynamic-quantized WebGPU path is ~6x slower
  despite the size reduction, so fp32 remains the performance default.
- CPU/WASM control is a valid fallback trade-off: fp32 WASM 5059.9 ms /
  2.23x versus INT8 MatMul WASM 3466.3 ms / 3.26x (exact oracle). Keep the
  candidate experimental and size/CPU-oriented; do not promote it to
  WebGPU without a different graph (e.g. static or weight-only) and new
  evidence.
- Reproduction/evidence: tools/scripts/convert_gigaam_rnnt_encoder_int8.py,
  tools/scripts/check_gigaam_rnnt_encoder_int8_parity.py,
  docs/reports/gigaam-rnnt-encoder-int8-probe-2026-08-30.md, and
  tools/data/results/gigaam/v3-rnnt-{encoder-int8-matmul-parity,
  int8-matmul-encgpu-decwasm-example,
  int8-matmul-encwasm-decwasm-example}.json.

CORRECTION (2026-08-30): Qwen decoder variants fp32/fp16/INT4 complete in
the browser; earlier "hang" conclusions were a harness defect.

- Evidence-integrity note: the syntax error in
  webgpu-agent-test/src/qwen-asr-webgpu.js (a stray brace introduced by the
  INT4 patch) prevented the page module from evaluating at all, so the page
  posted no payload and every leg timed out - that defect, not the
  artifacts or ORT, produced the "int4 hangs" / "fp16 hangs" conclusions
  and the nightly "still hangs" reading below. Found via a control leg:
  the fp32 composition went silent the same way, and vite reported a
  transform error. All harness pages now pass esbuild parse gates.
- Corrected browser results (Chrome, ORT Web 1.29, Blackwell, jfk 11 s,
  warmup + 3, exact fixed oracle):
  - fp32 decoder: median 5276 ms / 2.08x RTFx, stepAvg 101.8-105.3 ms;
  - fp16 decoder: median 5185 ms / 2.12x RTFx, stepAvg 101.9-103.6 ms;
  - INT4 MatMulNBits decoder: median 5106 ms / 2.19x RTFx, stepAvg
    97.1-102.5 ms.
- Interpretation: all three precisions complete and are precision-
  insensitive (~100 ms/step each today), so precision reduction is not a
  Qwen speed lever in this environment state. INT4 additionally carries
  real numerical drift on synthetic CPU probes (max abs ~20.4, cosine
  ~0.63) - keep it off by default. fp16 offers no benefit. No promotion.
- Environment warning (matches the Parakeet v3 check below): yesterday's
  fp32 control measured 1612 ms / 6.82x (stepAvg ~48-55 ms) on the same
  fixture and stack; today it is 5276 ms / 2.08x - a ~2.2x session-level
  swing. Cross-day RTFx comparisons for decode-loop-heavy models must
  cite capture date and re-run through the unified matrix.
- Still true: ORT 1.29.0 remains the pin (latest stable); the 1.30-dev
  nightly remains unpromoted (dev build, no demonstrated win - though its
  "still hangs" rejection is withdrawn as measured against the broken
  page). Real numerical drift rules INT4 out on quality grounds
  regardless of speed.

Qwen3-ASR decoder INT4 probe (2026-08-30, SUPERSEDED - see correction):

- Hypothesis: the 0.6B decoder step is weight-read-bound (30 steps x
  ~47-54 ms with 3 GB fp32 external weights matches an effective
  weight-streaming rate), so MatMulNBits INT4 weights should cut traffic
  ~4x. ORT Web 1.29 does ship MatMulNBits in the JSEP bundle (kernel
  registration plus 56 wasm references), and 196 of 253 MatMuls converted
  cleanly with lm_head excluded (1,479.7 MB total output).
- Result: the browser leg produced no result within three five-minute
  polls (no render, no posted payload) and was aborted - classify as
  HANGS_IN_BROWSER, per user direction deferred. The CPU parity probe was
  also inconclusive-negative: fp32-vs-INT4 logits showed max abs diff
  ~20.4 and cosine ~0.63 on a controlled step, though that probe's feed
  (zero/near-zero KV) is far outside real decoder state distributions, so
  treat the numbers as a red flag, not a quality verdict.
- Kept as opt-in tooling, off by default:
  tools/scripts/convert_qwen3_asr_decoder_int4.py,
  tools/scripts/check_qwen3_asr_decoder_int4_parity.py, and the harness
  ?decoderQuant=int4 override (scripts/run-qwen-webgpu.mjs
  --decoder-quant=int4). The qwen runner also persists error payloads now
  (writes a *-error.json) so future failures are not invisible.
- Revisit only with a new ORT Web release, a static-shape decoder export,
  or fp16-first evidence; do not retry the same conversion blind.

Qwen3-ASR decoder fp16 browser leg (2026-08-30, SUPERSEDED - see correction):

- The resolver already ships an fp16 decoder variant
  (decoder-*-fp16.onnx + 1.5 GB decoder-fp16.onnx.data) and the harness
  now exposes it via --decoder-dtype=fp16 / ?decoderDtype=fp16 (result id
  qwen3-asr-0.6b-official-webgpu-fp16dec). This composition had never
  been measured in the browser; it was the natural non-INT4 test of the
  decoder weight-bandwidth hypothesis.
- Result: no payload within the 15-minute runner timeout (no render, no
  posted result, exit 1), so the fp16 decoder does not complete on this
  Chrome/ORT 1.29/Blackwell setup - same practical outcome as INT4.
  Keep fp32 weights as the only completing browser decoder; recorded so
  the fp16 path is not retried blind on this stack.
- Qwen state after this round: argmax graph surgery rejected (reduction
  kernel cost), graph capture rejected (dynamic KV shapes), INT4 hangs,
  fp16 hangs. The remaining lever is a new artifact export (static cache
  shapes, fused kernels) or a newer ORT Web - not runtime tuning.

Qwen hang isolation and ORT 1.30-dev nightly probe (2026-08-30, SUPERSEDED - see correction; CPU isolation finding remains valid):

- Refined diagnosis: the fp16 decoder-step artifact is healthy on ORT CPU
  (one step in 0.33 s, fp16 logits argmax matches the fp32 reference on
  the same feed), so the browser hang is specific to ORT-Web's WebGPU/JSEP
  handling of this graph - not a broken artifact.
- Probed onnxruntime-web 1.30.0-dev.20260826 (latest nightly) with the
  1.29 tree backed up and fully restored afterwards: the Qwen
  --decoder-dtype=fp16 leg still produced no payload within 900 s
  (identical hang). The blocker is not fixed upstream as of this nightly;
  the production pin stays 1.29.0 stable (also confirmed still the latest
  stable on the registry).
- Candidate external action: file an upstream ORT-Web issue with the
  minimal repro (Chrome, decoder-step fp16 graph with dynamic past_len KV
  via session.run; completes on CPU, hangs in JSEP). Re-test the fp16 or
  INT4 decoder legs only after an upstream fix or a new stable release,
  using the same harness flags.

Unified all-models shared-window benchmark (2026-08-30):

- New orchestrator webgpu-agent-test/scripts/run-all-models.mjs runs every
  family on the shared 18.714 s LibriVox window (one warm-up + three
  measured runs; Whisper via a new en-greedy-gpu-kv-librivox matrix case)
  and emits one manifest plus a markdown table. X-ASR is excluded by
  design (deprioritized streaming-latency model).
- 2026-08-30 matrix: Whisper 27.45x (check), Parakeet v2 int8 hybrid
  37.15x normalized-exact, Parakeet v3 int8 hybrid 20.44x exact,
  GigaAM CTC 90.11x, GigaAM RNN-T 32.41x, SenseVoice 46.79x. Qwen posted
  no payload within 15 minutes on this window and is recorded as
  artifact-fragile beyond the 11 s fixture. [CORRECTION: the Qwen "no
  payload" cause was the harness page defect (see the correction section
  above), not the artifact; re-measure Qwen on this window with the fixed
  page before drawing window-length conclusions.]
- Two methodology catches from the first bring-up: (1) the Parakeet
  runner defaults to the full-WebGPU decoder composition - the hybrid
  requires --mode=wasm, and running the wrong one initially looked like a
  4x regression; (2) the INT8 decoder on the full-WebGPU composition
  emits garbled transcripts, so INT8 stays transcript-safe only on WASM
  (library default already correct).
- Evidence: docs/reports/all-models-shared-window-benchmark-2026-08-30.md
  and tools/data/results/cross-model/all-models-shared-window-2026-08-30.json.

Parakeet v3 20x-vs-37x reproducibility check (2026-08-30):

- Paired same-session probes of the v3 int8 hybrid on the shared window are
  all exact and all land at 20-23x: ONNX preprocessor 955.7 ms / 19.70x,
  JS preprocessor 828.5 ms / 22.76x, 1-thread WASM 892.9 ms / 21.05x,
  12-thread 923.0 ms / 20.44x. Preprocessor and thread count are ruled out;
  v2 reaching 37.15x today rules out a machine-wide slowdown. The 37.2x
  2026-08-29 v3 record is therefore an environment-dependent number
  (Chrome/driver/thermal drift suspected), not a library regression.
- Methodology note: single-session records for decode-loop-heavy models
  should cite the capture date and be re-verified through the unified
  matrix before being used as promotion baselines.

Whisper stable-beam GPU-KV implementation (2026-08-30, SHIPPED):

- Beam 2 measures 5.57x vs greedy 27.02x on the same fixture. Attribution
  complete: GPU-KV hard-rejects beam search (assert in
  src/models/whisper-seq2seq/executor.ts before the runGreedyGpuKvDecode
  dispatch), so beams run the snapshot-based CPU-KV path. Every beam step
  round-trips the whole KV to CPU: copyAndReleaseWhisperPresentKv copies
  gpu buffers into plain snapshots and releases them, then runStep
  re-materializes fresh tensors via cloneDecoderKvDataForInput. Cost grows
  with sequence length and explains ~48 ms/step x 98 executions vs
  greedy's ~14 ms x 49 on GPU-resident KV.
- Implemented as designed: runInit/runStep callbacks pass gpu-buffer
  present tensors through without copy/release (prepared-past path of
  runDecoderStepSplit), a generation tracker disposes any tensor not
  fed/produced for numBeams+1 generations and everything on decode end,
  decoder init/step sessions wire createWhisperGpuKvOutputLocation when
  the flag is on, greedy and CPU-KV beam paths untouched, batching
  disabled while the flag is active. Core beam logic needed zero changes:
  caches already pass through opaquely with read-only sharing.
- Validated: GPU-KV beam-2 transcript byte-identical to the back-to-back
  CPU-KV stable beam-2 control (50 tokens); GPU beam 1585-1938 ms
  (~17.7-18.9x RTFx) vs CPU beam 5417.8 ms (5.52x) - a 3.2-3.4x speedup
  for beam-2 quality; three soak pages without double-free or OOM; abort
  propagates through the same finally-disposal block.
- Config: source option experimentalGpuKvBeam: true TOGETHER with
  experimentalGpuKvCache: true and a WebGPU decoder backend (the beam flag
  is a modifier of the GPU-KV cache flag); temperature sampling and
  best_of stay rejected; experimentalBatchedBeam is ignored while the
  GPU beam flag is active. Harness: ?gpuKvBeam=1 and matrix case
  en-stable-beam-2-gpu-kv.
- Debug lesson recorded in the report: reading a gpu-buffer tensor's
  .data getter throws even for an instanceof probe - branch on
  tensor.location before any .data access.
- Evidence: docs/reports/whisper-beam2-gpu-kv-2026-08-30.md and
  tools/data/results/whisper/beam2-{gpu-kv,gpu-kv-soak,gpu-kv-first-pass,
  cpu}-jfk-30s.json.

Whisper 4-graph revalidation on ORT Web 1.29 (2026-08-30):

- Greedy + GPU-KV on the 30 s JFK clip measures 27.02x RTFx (1106.6 ms
  total: 183.2 encode, 707.1 decode over 49 steps) - the documented ~26x
  target holds on the current stack; no regression from the ORT upgrade.
- Stable beam 2 measures 5370.4 ms / 5.57x with the identical 50-token
  sequence - roughly 2x faster than the recorded 2026-08-23/25 baselines
  (2.74-2.86x), reflecting decoder optimizations landed since.
- The remaining recoverable warning is whisper.decoder-align-legacy
  (decoder_align export marker; timestamp interpolation fallback) - an
  artifact re-export item, not a runtime defect. CLOSED for the runtime side:
  the causal r5 artifact now passes the browser harness with zero warnings
  and identical throughput (see the causal decoder-align slice below).
- Evidence: docs/reports/whisper-4graph-ort129-revalidation-2026-08-30.md
  and tools/data/results/whisper/*revalidation-2026-08-30.*

Whisper causal decoder-align browser validation (2026-08-30):

- The r5 causal family (N:\models\whisper-large-v3-turbo-causal-fp16-20260825-r5)
  is a first-class persistent route in the browser harness (/models/fp16-causal/;
  presets fp16io-causal-webgpu and causal-r5-full-webgpu). Timestamped runs on
  the causal decoder_align produce wordAlignmentSource=fast, first-word anchor
  In 2.12-2.84 s (faster-whisper ref 2.16-2.84 s), and an EMPTY warnings array
  on all five cells (greedy GPU-KV, stable beam 2, batched beam 2, r5-full,
  30 s greedy at 27.02x - identical to the no-timestamps revalidation).
- Remaining boundary is artifact publishing only: the public HF preset still
  ships the legacy align graph, so default downloads keep the recoverable
  fallback warning until an approved re-export (audit_publish.py gates).
- Evidence: docs/reports/whisper-causal-decoder-align-browser-2026-08-30.md
  and tools/data/results/whisper/*causal*-2026-08-30.json

X-ASR speculative batched joiner decode (2026-08-30):

- Greedy RNNT decode of one chunk's frames against an unchanged decoder
  state is row-parallel (joiner graph leading dim is N: encoder_out
  [N,512], decoder_out [N,512], logit [N,5000]), so the per-frame joiner
  runs are replaced by one speculative batch run: blank rows never change
  the decoder state, the first non-blank row is exactly the sequential
  emission, and the frame suffix is re-batched after each token. Output is
  token-for-token identical by construction.
- Sticky fallback: a batched run that throws or returns non-row-parallel
  shapes latches batching off for the executor and the identical
  sequential path takes over mid-chunk. Abort semantics unchanged
  (PipelineAbortedError propagates, caller state tensors stay intact).
- Chrome headless WebGPU (Blackwell, 11.29 s jfk-short, 55 chunks,
  warmup+3 runs): joiner dispatches per pass 311 -> ~88; median
  transcribeMs 8981 -> 8336 (RTFx 1.22 -> 1.32; 7716/1.43 in the profiled
  session; +-10% cross-session variance applies). Transcript byte-identical.
- Phase profile (new --profile harness mode, 2 passes): encoder 136 runs
  14,983 ms (~110 ms/run), decoder 188 runs 757 ms, joiner 175 runs
  626 ms. Streaming is now ~90% encoder-bound; the joiner win is real but
  bounded. Next X-ASR lever: attribute the ~110 ms stateful Zipformer2
  encoder run (dispatch vs 116-tensor GPU state I/O vs kernels) before
  touching chunk-window sizes.
- Evidence: docs/reports/x-asr-joiner-batching-2026-08-30.md,
  tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-stream{,-profiled}-chrome.json.
  Validation: tests/x-asr-joiner-batching.test.ts (4), real-artifact Node
  WASM + streaming parity (XASR_ONNX_SMOKE=1, 3). Full suite green:
  1037 passed / 18 artifact-gated skips.

X-ASR backend placement after batching (2026-08-30, deprioritized):

- The Chrome harness gained an explicit --backend=wasm control (WebGPU
  stays the default). On the 11.29 s / 55-chunk JFK fixture both paths are
  exact: WebGPU 8336 ms / 1.3196x RTFx; all-WASM 6822 ms / 1.6124x RTFx
  (~18% faster on this Blackwell/ORT 1.29 host). Profiled two-pass
  encoder totals: WebGPU ~110.17 ms/run vs WASM ~97.54 ms/run; the
  encoder dominates both backends.
- Per user direction, X-ASR is deprioritized: it is a niche streaming
  model whose README advertises streaming latency, so RTFx is not its
  native metric. If revisited, measure per-chunk (algorithmic) latency
  and first-token latency alongside throughput; until then no further
  optimization cycles.
- Evidence: docs/reports/x-asr-backend-placement-2026-08-30.md and
  tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-wasm-stream
  {-profiled,}-chrome.json.

SenseVoice fp16 CTC decode hot path (2026-08-30):

- argmaxAndSelectedLogProbsFp16 (src/ctc/decoder.ts, exported via
  src/ctc/index.ts) decodes raw float16 logit bits directly: integer
  ordering-key max scan + a lazily built Float64Array(65536) exp lookup
  table, with score best - log(sum) algebraically identical to the
  reference formulation. Per-row parity fallbacks (NaN/inf codes,
  maxima outside the [-80, +80] exp safe zone, truncated rows) reroute
  through the converting generic pipeline, so raw-logit graphs keep
  identical semantics.
- SenseVoice executor routes single and batch paths through
  decodeLogitsBlock: fp16 tensors stay Uint16Array bit subarrays (no
  float materialization, zero-copy per batch row); fp32 tensors keep
  the generic pipeline.
- Node microbenchmark (same-launch, realistic bits): 187 x 25055
  149.7 -> 13.1 ms (11.4x); 316 x 25055 260.0 -> 22.0 ms (11.8x);
  parity gate ids equal, max score diff 0.00e+0.
- Chrome headless WebGPU (Blackwell, warmup+3 runs, 18.7 s librivox):
  decodeMs 385.6 -> 163.8-178.8, median-run RTFx 26.42 -> 43.1-48.7,
  transcript byte-identical (59 tokens). Evidence:
  tools/data/results/sensevoice/sensevoice-small-librivox-18s-warmed{-fp16-lut,}-webgpu-chrome.json
  and docs/reports/sensevoice-fp16-decode-hotpath-2026-08-30.md.
- Validated by tests/ctc-decoder-fp16.test.ts (5 parity tests) and a 6-shape
  adversarial Node probe (wide +-60000 logits, safe-zone-edge tails,
  all-zeros, denormal soup, truncated buffer): ids equal, maxDiff 0.
  Full suite green: 1033 passed / 18 artifact-gated skips.
- GigaAM CTC browser graphs emit fp32 logits today, so no wiring change
  was needed there; the fast path is available if fp16 logits ship.

Whisper frontend mixed-radix FFT slice (2026-08-30):

- The exact n_fft=400 mel path (Whisper default and the Qwen frontend reuse)
  previously ran Bluestein chirp-z: three 1024-point radix-2 FFTs per frame for
  a size that factors cleanly as 25 * 16. New src/audio/mixed-radix-fft.ts
  implements an exact N = 5^a * 2^b Cooley-Tukey decomposition (recursive
  radix-5 down to a cached radix-2 kernel), and src/audio/whisper-mel.ts
  selects it whenever isMixedRadixSize(nFft) holds, keeping BluesteinRfft as
  the fallback for uncovered sizes.
- Node mel bench (node tests/smoke/benchmark-whisper-mel.mjs --runs=7
  --mels=128): 30 s exact-400 avg 173.0 ms -> 94.1 ms (RTFx 173.5 -> ~319;
  ~1.7-1.8x). 1 s 7.2 -> 4.6 ms, 10 s 56.6 -> 30.6 ms. The experimental 512
  path is untouched (~50 ms). Evidence:
  tools/data/results/whisper/whisper-mel-400-fft-before-bluestein-2026-08-30.json
  and whisper-mel-400-fft-after-mixedradix-2026-08-30.json.
- Correctness first: naive-DFT oracle max abs error 3.8e-11 across N in
  5..1600 (incl. 400/200); transformRealInput(400) error 1.35e-12;
  tests/mixed-radix-fft.test.ts locks oracle parity, Bluestein agreement at
  200/400, zeroImaginary equivalence, and RangeError rejection. Full suite
  green: 1028 passed, 18 skipped; typecheck clean.
- Browser A/B control run deliberately skipped: the expected end-to-end saving
  is ~78 ms of the 1106.6 ms 30 s greedy GPU-KV run (~7%), inside the
  documented +/-11% cross-session variance, so a same-launch A/B would not be
  honest evidence. The mel code is identical JS in browser and Node.
- Future consolidation candidate (not done this slice to minimize risk):
  RadixFivePowerOfTwoFft in src/models/lasr-ctc/mel.ts could be replaced by
  MixedRadixFft, which strictly generalizes it (5^a * 2^b vs 5 * 2^m); no
  current model uses the extra 25 * 2^m coverage there.

Browser revalidation evidence commit (2026-08-30):

- Committed `aea2029` records the four verified warmed Chrome/WebGPU runs from
  the artifact revalidation pass: GigaAM CTC 42.74x and SenseVoice 26.42x on
  the shared 18.7 s clip (throughput-only, `qualityOracle: none`), and the
  GigaAM RNN-T default-hybrid (28.74x) vs all-WebGPU (2.90x) compositions on
  the official Russian `example.wav` with exact parity. Backup branch
  `backup/pre-evidence-commit-revalidation-2026-08-30`.
- Three working-tree files were restored to HEAD instead of committed: their
  contents mismatched their filenames (a WASM SenseVoice run had overwritten
  `sensevoice-small-jfk-short-webgpu-chrome.json`; the X-ASR webgpu-chrome file
  held a LibriVox-clip run against the JFK oracle, marked
  `PREPROCESSING_MISMATCH`; the GigaAM jfk file duplicated a rerun already
  evidenced elsewhere). Lesson: harness result paths must be derived from the
  actual run config (backend + audio + oracle), not only from CLI defaults.

- Experimental family descriptors are clone-safe, and all model families now
  share session release, abort, and dispose coverage; disposing a model no
  longer double-releases ORT sessions during `runtime.dispose()`
- In-flight browser transcription can be aborted without killing the worker,
  so a loaded model stays loaded (`e8624e6`, `b8dc570`, `f657522`,
  `9bd6fd0`, `bacab6e`)
- Artifact-gated real-artifact WASM rerun (`3de0bd7`): GigaAM CTC 4/4
  (fp32, fp16, mixed-length batch), SenseVoice 3/3 (WASM, batch), X-ASR 3/3
  (WASM + public stateful streaming), GigaAM RNN-T 2/2 (WASM)

X-ASR incremental frontend slice (2026-08-29):

- Streaming `pushStream()` no longer reruns the complete accumulated waveform
  through the 80-bin Kaldi-compatible fbank frontend on every chunk. The
  family-specific frontend now processes only newly sample-backed frames and
  keeps a bounded 400-sample raw tail; reflected right-edge frames are held
  until the next chunk or `final=true` so full-buffer semantics remain exact.
- Deterministic parity across uneven chunks is exact (`maxAbs=0`). The
  reproducible Node CPU microbenchmark shows 4.4933x lower combined
  frontend/storage wall time at 2 seconds (22.8828 -> 5.0927 ms) and 23.6110x
  lower time at 10 seconds (543.0221 -> 22.9987 ms), using 200 ms chunks and
  three timed runs after one warm-up. The separate frontend-only controls are
  4.5512x and 17.5392x faster; the baseline uses exact cumulative copies while
  the candidate uses amortized capacity growth. These are frontend/storage
  results, not end-to-end RTFx claims.
- Evidence and rerun command: `docs/reports/x-asr-incremental-frontend-benchmark-2026-08-29.json`
  and `npm run benchmark:x-asr-frontend -- --runs=3 --durations=2,10 --json`.
- Real-artifact browser checkpoint also passes: Chrome headless/WebGPU on
  NVIDIA Blackwell, ORT Web 1.29.0, 55 x 200 ms streaming chunks, and the
  exact 55-token X-ASR oracle (`8,981.14 ms`, `1.2248x` RTFx). This validates
  the new frame-boundary behavior in the actual encoder-cache path; it is not
  a browser before/after speed claim. Evidence:
  `docs/reports/x-asr-webgpu-streaming-parity-2026-08-29.json`.
- The executor retains the exact logical cumulative audio view for stream
  duration metadata while reusing an amortized backing buffer; retained
  capacity can exceed logical length until stream disposal.

Earlier streaming and validation slices:

- High-level owned streaming exposed on loaded handles (`a1264ca`,
  `db6e709`) and divergent overlapping-window text merge fixed
  (`a0818f1`, `3ea33b6`)
- Fake-microphone browser smoke probe and acceptance evidence
  (`c19014a`, `27fdc7c`); high-performance WebGPU adapter preference
  (`97139dc`); Node WebGPU routed through `onnxruntime-native`
  (`7ef50d2`)

## Ongoing principles and direction

- Reuse well-designed library tools and design within `speech-recognition`
  itself
- Create the most effective ONNX graphs (like Whisper Large V3 Turbo)
- Use the best backend combination when needed: encoder on WebGPU, decoder on
  WASM
- Optimize the newly ported families (GigaAM, SenseVoice, X-ASR, Qwen):
  current throughput is low and may be ORT CPU/WASM-bound; reproduce with
  real WebGPU in the browser before changing code, then apply the hybrid
  backend composition where measurements justify it. GigaAM RNN-T now exposes
  explicit encoder/decoder/joiner provider overrides so this comparison can be
  made without changing the all-one-backend default.
- Treat incremental frontend/state work as a required model-specific pass for
  streaming families: establish frame-boundary semantics first, delay unstable
  reflected frames, retain only bounded context, and measure parity and CPU
  savings before touching encoder placement.
- Validate WebGPU work with the existing Chrome headless real-WebGPU browser
  smoke harness (the same approach used for the Whisper split-graph port);
  ORT-node fp16 binding support is not urgent and not mandatory
- Update `onnxruntime-web` dependencies only carefully; keep the working
  pin until a reproducible upgrade passes the full browser matrix
- Frontend/mel feature extraction is CLOSED for optimization purposes unless a
  new measurement shows it is >15% of a model's end-to-end time. The exact
  n_fft=400 path is now a mixed-radix FFT at ~319x RTFx in Node (~7% of the
  Whisper 30 s browser run), the 512 experimental path is ~595x, and the
  GigaAM 320 path already got its radix-5 pass. Spend optimization effort on
  encoder/decoder graph execution, backend placement, and lifecycle instead.
- Experimental implementations remain clearly marked until the completion
  gates in this document are satisfied
- Single-package ESM-first speech-focused headless framework-neutral runtime
- Mission priorities: public API/DX, runtime reliability, verified ASR
  models, WebGPU/WASM perf, realtime/streaming, examples/docs/benchmarks,
  reusable porting/parity infrastructure
- Respect folder boundaries per `AGENTS.md` and `docs/architecture.md`

## Model-specific performance optimization (primary directive)

For every supported or newly ported ASR model, working correctly is the
starting line, not the finish line. After correctness is verified,
systematically investigate optimization opportunities and keep going until
measurements show the remaining bottlenecks are not worth the engineering
cost. Correctness first, optimization second, measurement throughout.

Investigate, in rough priority order:

- WebGPU execution and ONNX Runtime Web/WebGPU EP behavior for the exact
  artifact and browser;
- graph structure and unnecessary CPU↔GPU synchronization or tensor
  transfers (keep state GPU-resident; read back the minimum: one token id,
  not a full logits vector);
- preprocessing, encoder, decoder, joiner, and postprocessing bottlenecks,
  attributed by measured phase shares before choosing what to optimize;
- audio preparation parity (source-rate decode, channel mixing, resampling,
  and feature lengths) before changing model math; browser AudioContext
  resampling can change tokens even when duration and backend are unchanged;
- incremental computation, reusable caches, KV/state/cache reuse, and
  redundant computation across streaming windows;
- tensor allocation, copying, disposal, and memory lifetime in hot loops;
- JavaScript/TypeScript hot paths (allocation-free step loops, precomputed
  session input/output names, bounded top-k instead of full sorts);
- WASM vs WebGPU placement per component; backend choice is
  workload-specific and must be re-validated per artifact;
- model initialization, warm-up, load size, and startup latency;
- WebGPU dispatch overhead and command-buffer behavior;
- ONNX graph surgery (strip unnecessary casts, patch dynamic dimensions,
  append argmax, fuse or restructure hot subgraphs) when it unlocks a
  measured win;
- precision reduction and mixed precision (fp16 I/O, pure fp16, int8, int4);
  quantize where it usefully reduces VRAM/memory/size without unacceptable
  accuracy or speed regressions, and never assume quantization is faster —
  benchmark it;
- VRAM and system-memory usage as first-class metrics.

Maintain before/after benchmarks for every optimization: RTFx/latency,
memory or VRAM where measurable, model size, initialization time when
relevant, and recognition-quality regressions. An optimization without a
before/after measurement is not an optimization; it is a hypothesis.

### Reference case studies (study before optimizing a new model)

Whisper Large V3 Turbo — RTFx 4.8× → 27.6× (30 s audio in ~1.1 s). Sources:
`docs/Whisper-Optimizations.md` (14 documented pitfalls),
`docs/OPTIMIZATION-SPRINT-REPORT.md`, merge `c1f50ce`. What produced the
jump:

1. GPU KV cache bridge — decoder state stayed in gpu-buffer tensors across
   steps; decode phase 5.7×;
2. stripped fp16 encoder graph — removed the input Cast by exporting fp16
   directly; encode phase 6.9×;
3. fast mel — power-of-2 FFT path; preprocess 2.9×;
4. scalar-only readback and scalar beam: immutable encoder KV shared and
   broadcast across beams, bounded top-k, no per-beam full-vocabulary
   log-softmax;
5. hot-loop hygiene: precomputed KV input names, allocation-free step loop,
   async model loading with overlapped metadata.

Negative results to remember: the WebGPU 'simple' buffer-cache mode regressed
RTFx ~12% and was reverted; self-speculative decoding breaks token parity and
needs a real draft model (the multi-token decoder-step graph is kept as
backward-compatible infrastructure); graph capture only helps multi-chunk
audio; pure-JS mel was already good enough to defer WASM/WebGPU mel.

Parakeet TDT v3 — the proven hybrid composition: encoder on WebGPU, decoder
and joint on WASM, per-component backend selection as a library capability.
int8 WASM runs at RTFx ~4.6× with roughly half the memory (baseline report
`docs/reports/parakeet-tdt-v3-local-baseline-2026-08-26.md`). The decoder was
kept on WASM because the WebGPU EP lacked GRU/LSTM support; ORT Web 1.29.0
added GRU and LSTM to the WebGPU EP, so the decoder-on-WebGPU composition must
be re-probed per artifact instead of assumed impossible. The same
GPU-resident-state technique produced the X-ASR streaming win (3.46×) recorded
above. The 2026-08-29 full-model probe adds a reusable warning: compare
browser audio preparation against the native/reference resampler first. The
default target-rate AudioContext path stopped at 74 tokens, while deterministic
WAV parsing plus linear resampling restored exact 91-token parity and 18.66x
RTFx with the same fp16/WebGPU encoder composition.

Turn recurring lessons into reusable tools, tests, fixtures, and playbooks
(`tools/model-debugging/playbooks/`, stage comparator, benchmark harness) so
each new port inherits the methodology automatically. Keep refining the
methodology as new ORT behaviors, graph patterns, and bottlenecks are found.

Shared short-window benchmark and RTFx regression probe (2026-08-29):

- The browser benchmark contract now uses the same deterministic
  `librivox.org.wav` window (18.714 s, SHA-256
  `2F6886C1956765B56B996BD6FBB00C5E4001368DB8CB4BE94987CA2E3166B8B4`) for
  cross-model throughput, performs one same-session warm-up, and reports
  warm-up separately from measured repetitions. Correctness remains a separate
  labeled oracle; unlabeled runs carry `qualityOracle: null` and must not be
  treated as WER evidence. Keep all shared benchmark windows below 30 s.
- Parakeet runs on the shared window preserve v3 exact text parity (fp16
  WebGPU encoder + fp32 WASM decoder, ONNX preprocessor) and v2 normalized
  parity (fp16 WebGPU encoder + int8 WASM decoder, JS preprocessor). Current
  warmed medians are approximately 18.1x and 28.2x native RTFx respectively;
  the historical Parakeet.js v2 benchmark on 15–30 s samples reports a median
  of 80.71x with fp32 encoder/int8 decoder and 11-thread-class browser
  settings. This is a real configuration/runtime regression candidate, not a
  short-audio explanation alone.
- Threaded-WASM worker path root cause fixed in the sibling Chrome harness
  (2026-08-29): `crossOriginIsolated` was already true, but the `/ort-dist/*`
  static responses only sent `Cross-Origin-Resource-Policy`; ORT's module
  workers need their sibling `.mjs`/`.wasm` subresources to also carry
  `Cross-Origin-Embedder-Policy: require-corp`, otherwise Chrome blocks them
  with `coep-frame-resource-needs-coep-header` and the probe never completes.
  After the header fix, `cpuThreads=4` completes with exact 91-token parity but
  is measurably slower than single-thread for this GRU decoder (about 14.2x vs
  17.7x warmed RTFx, decoder 950 ms vs 720 ms), so the production Parakeet
  default stays single-threaded WASM until a larger decode workload justifies
  threads. The benchmark harness now omits `cpuThreads` by default so it uses
  the library's `navigator.hardwareConcurrency` value; explicit
  `--cpu-threads=N` remains the diagnostic control.
- The 146.326 s JFK source is retained for provenance, but the Parakeet v3
  encoder rejects its 1830-frame unwindowed input (static limit 1024). Use
  supported sub-30-second clips or a verified long-audio windowing pipeline;
  never report that shape failure as throughput.
- The same warmed shared-window Whisper control is measurable at about 12.4x
  native RTFx with CPU KV; the GPU-KV candidate is about 8.4x on this clip and
  is therefore a workload-specific negative result, despite its earlier JFK
  win. Keep per-model/backend placement decisions evidence-driven.
- Evidence and exact commands: `docs/reports/cross-model-audio-window-benchmark-2026-08-29.md`
  and its JSON manifest.

## Current WebGPU EP optimization task

ORT Web was upgraded 1.27-nightly → 1.29.0 stable and validated end-to-end
(see Completed below). The built-in WebGPU EP exposes the GRU/LSTM support used
by the decoder spike. Separately, the native ONNX Runtime WebGPU Plugin EP
0.3.0 release adds GRU and DFT support, deferred dispatch for parallel shader
compilation, configurable GPU-buffer caching, improved FP16/MatMul kernels,
and broader integer coverage. Treat those as two different compatibility
surfaces. Work items, in order:

1. [Completed 2026-08-29] Probe the Parakeet TDT decoder/joint graphs on the
   built-in WebGPU EP in Chrome via an opt-in decoder-backend override. The
   corrected controlled A/B and the library-entry GPU-state A/B are recorded
   below. The `decoderStateOutputLocation` path is opt-in and preserves the
   default encoder-WebGPU/decoder-WASM composition until cross-adapter
   lifecycle evidence justifies promotion.
2. [Qwen placement/profile sub-slice completed 2026-08-29] Measure
   per-component placement again for every family on 1.29.0 — the X-ASR
   WASM-vs-WebGPU and buffer-cache conclusions may shift with the new EP;
   re-benchmark instead of inheriting old verdicts. Qwen's corrected browser
   entry-point A/B and decoder phase profile are recorded above; remaining
   family measurements and cross-browser repetition stay open.
3. [Research boundary] Run a bounded compatibility spike against the separate
   native WebGPU Plugin EP 0.3.0 (Python/.NET, not npm) only when the plugin is
   available locally. Compare Parakeet decoder/joiner session creation,
   first-run, warm-step latency, memory, finite logits, vocabulary-sliced token
   parity, and disposal against the built-in ORT Web 1.29 path. Do not add the
   plugin as a browser dependency or block library work on its availability.
4. [Qwen graph-capture and ArgMax probes completed 2026-08-29; promotion remains open]
   Promote only lifecycle-safe, end-to-end-proven state/cache
   placement. Track tensor ownership and disposal explicitly; a decoder-only
   GPU-state win is not sufficient for production promotion. For each model,
   use the phase telemetry to select the next bottleneck and record rejected
   low-yield hypotheses instead of optimizing unmeasured hot paths. Keep
   graph-surgery candidates reversible and require provider-specific evidence;
   ArgMax/readback reduction is not a win when the added reduction kernel
   dominates execution.

Local availability check (2026-08-29): the separate native Plugin EP 0.3.0 is
not installed on this host. Python ONNX Runtime exposes TensorRT/CUDA/CPU only,
and the Node workspace contains the built-in `onnxruntime-web` 1.29.0 plus the
existing nightly `onnxruntime-node`; no plugin package or .NET project is
available. Do not block browser optimization on this research boundary.

Probe status (2026-08-29, corrected controlled browser A/B):

- Native WebGPU probe (`onnxruntime-node` wgpu adapter): the Parakeet v3
  `decoder_joint-model.onnx` GRU graph RUNS on a GPU EP. Session creation
  479 ms, warm steps 7–9 ms, outputs finite, logits dims `[1,1,1,8198]`
  (8193 vocab + 5 TDT duration logits — argmax must slice to vocabSize). The
  graph itself is WebGPU-fast; GPU decode is a real option, not a Wasm-only
  constraint.
- Browser spike harness built in `webgpu-agent-test`:
  `parakeet-tdt.html` + `src/parakeet-tdt-webgpu.js` +
  `scripts/run-parakeet-tdt-webgpu.mjs` (full-model and decoder-only spike
  modes). Asset routes `/parakeet-v3/` and `/parakeet-audio/` added.
- Corrected browser A/B: on a real NVIDIA Blackwell adapter, both ORT Web
  `1.27.0-dev.20260506-673c3320fc` and stable `1.29.0` completed the
  decoder-only GRU spike on both WebGPU and WASM. Three corrected runs per
  version/backend passed with finite outputs and identical vocabulary-sliced
  token IDs (`8192`); the raw argmax was duration logit `8194` in the
  `[1,1,1,8198]` output, so the probe now limits token selection to the 8193
  vocabulary entries. Representative medians were WebGPU session-load
  `1.41 s` (1.27) vs `1.46 s` (1.29), with warm-step means varying from
  `14.0–32.2 ms` (1.27) and `9.9–14.9 ms` (1.29); WASM warm-step means were
  `5.5–5.8 ms` (1.27) and `5.5–6.5 ms` (1.29). This disproves the earlier
  uncorrected “1.29 session-create hang” classification.
- The spike harness was repaired before accepting that evidence: spike mode
  no longer imports the full library before the direct ORT probe, its result
  logger is defined, and the runner recognizes its two per-EP result records.
  These fixes are validation infrastructure only and do not change the
  library's default backend composition.
- Opt-in 1.29 GPU-state experiment: `preferredOutputLocation` kept both GRU
  state outputs on `gpu-buffer` and preserved the same five-step vocabulary
  argmax sequence. Three WebGPU runs had warm-step means `9.01`, `8.26`, and
  `7.78 ms`, versus `12.73–13.14 ms` across comparable CPU-state baselines
  (about 29–41% lower in this decoder-only microbenchmark). This is a hypothesis for
  the library's next Parakeet optimization slice, not an end-to-end promotion:
  full-model transcript parity, repeated-transcription memory behavior, and
  safe GPU-tensor disposal remain unverified.
- Model-specific decode hot path (library, 2026-08-29): TDT now borrows
  float32 logits views before synchronous disposal, and the shared
  `confidenceFromLogits` helper skips entropy work that its API does not return.
  The 10,000-run Node harness measured `0.0849 ms` p50 for confidence-only
  versus `0.3970 ms` p50 for full TDT-vocabulary quality (about 79% lower), and
  `0.0928 ms` p50 for borrowed logits versus `0.1114 ms` for the copy control
  (about 17% lower). This is post-processing evidence, not an end-to-end RTFx
  claim; full-suite correctness remains required.
- Open result: the corrected full-model browser probe reaches model-ready and
  shows large WebGPU-encoder latency gains. Target-rate AudioContext runs fail
  transcript parity, while the deterministic native-rate WAV path now passes
  exact parity at 18.66x RTFx for one fp16/WebGPU composition. Repeat that
  result across browsers and artifact variants, and diagnose external-data
  mounting/provider behavior before changing any preset default. Until exact
  parity and end-to-end GPU-state lifecycle proof exist, keep the production
  default `encoder-WebGPU/decoder-WASM` composition.
- Lifecycle refresh (2026-08-29): the corrected all-WebGPU Parakeet browser
  harness reproduced the exact 91-token transcript for 3/3 native-rate runs
  (median 3,420.915 ms / 5.477x RTFx) and completed model disposal; the
  explicit GPU-state replacement/disposal diagnostic also passed all five
  steps twice without the historical `null function` failure. Treat this as
  refreshed evidence, not a promotion: the opt-in library state-location path
  is now wired through hub, local, and direct Parakeet loading, but full-model
  parity, repeated transcriptions, and disposal must still be repeated on more
  than one browser/adapter before changing the default. Evidence:
  `docs/reports/parakeet-tdt-v3-webgpu-lifecycle-refresh-2026-08-29.json`.
- Library opt-in state A/B (2026-08-29): the new `decoderStateOutputLocation`
  path was exercised through `loadSpeechModel` on the same Chrome/NVIDIA
  Blackwell fixture. GPU-state output preserved the exact 91-token transcript
  for 3/3 runs and reduced the warm median from `3,474.620 ms` (`5.3918x`
  RTFx) to `2,871.875 ms` (`6.5244x` RTFx): `17.3471%` lower latency and
  `21.006%` higher RTFx, with comparable JS-heap snapshots and no disposal
  error. This is strong single-adapter evidence for an opt-in optimization,
  not a default change; repeat on another browser/adapter and run a longer
  lifecycle soak before promotion. Evidence:
  `docs/reports/parakeet-tdt-v3-library-gpu-state-ab-2026-08-29.json`.
  The same host exposed no adapter under Chrome Vulkan or D3D12 ANGLE, so those
  secondary checks are recorded as `WEBGPU_NO_ADAPTER` rather than treated as
  performance failures.

- ORT 1.29 storage-cache sweep (2026-08-29): explicit `bucket`, `simple`,
  `disabled`, and `lazyRelease` modes all preserved exact Parakeet TDT
  transcripts on the same Chrome/NVIDIA fixture, but no cache-only mode beat
  the bucket control repeatably; `simple` plus GPU-state was also slightly
  slower than the prior bucket-default GPU-state run. The library exposes the
  knobs for model-specific experiments while retaining ORT defaults. Evidence:
  `docs/reports/parakeet-tdt-v3-webgpu-cache-sweep-2026-08-29.json`.
- Five-run Parakeet GPU-state soak (2026-08-29): the same library-entry
  fixture preserved exact 91-token parity for all five CPU-state and
  `gpu-buffer`-state runs. GPU-state reduced median warm latency from
  `3,491.575 ms` to `2,907.200 ms` (`16.7367%`) and improved median RTFx from
  `5.3655x` to `6.4454x` (`20.1267%`); load time was also `7.3665%` lower in
  this repeat. The strengthened harness records model/runtime disposal errors
  and found none, but the final heap sample drops sharply in both controls;
  retain the path opt-in pending another browser/adapter.
  Evidence: `docs/reports/parakeet-tdt-v3-webgpu-lifecycle-soak-2026-08-29.json`.
  A Chrome SwiftShader attempt also returned `WEBGPU_NO_ADAPTER`; the host
  therefore still lacks a second usable adapter for the promotion gate.

- [Completed 2026-08-29] GigaAM RNN-T phase profile and provider matrix: the
  official v3 E2E artifact and
  captured waveform pass exact 78-token parity on Node WASM. Three measured
  runs attribute roughly 93% of transcribe time to the encoder (5.25–5.61 s),
  versus about 3–5% to decoder/joiner work (0.19–0.26 s) and 1–3% to JS
  preprocessing. A frame-extraction/tensor-hoist candidate was rejected after
  its 3-run median moved from 6.158 s to 6.105 s but p90 regressed from 6.171
  s to 6.593 s; this is not a promotion-quality win. The next bounded test is
  real Chrome WebGPU encoder + WASM decoder/joiner; it measured 1,910 ms / 5.92x
  RTFx, versus 4,638 ms / 2.44x all-WebGPU and 5,948 ms / 1.90x all-WASM;
  all three were exact on NVIDIA Blackwell with ORT Web 1.29.0. The matrix is
  in `tools/data/results/gigaam/gigaam-rnnt-browser-component-matrix-2026-08-29.json`.
  Same-session repeats warmed the hybrid path to a 455 ms median / 24.87x RTFx
  (all exact), while all-WebGPU and all-WASM medians were 2.64x and 1.98x.
  Repeat across browsers and sessions before changing a preset default.

- [Profile 2026-08-29] GigaAM multilingual CTC now reports separate
  preprocessing, ORT encoder, and CTC readback/decode phases instead of
  attributing the entire call to `encodeMs`. On three fresh Chrome/NVIDIA
  Blackwell runs, the warm median was `363.755 ms` (`31.8025x` RTFx):
  preprocessing `74.295 ms` (20.42%), encoder `283.050 ms` (77.81%), and
  decode/readback `2.820 ms` (0.78%), with exact JFK parity in every run.
  This rejects CTC tensor-copy/argmax surgery as a high-value target for this
  family; future work should target the encoder graph/provider
  behavior. Evidence: `docs/reports/gigaam-ctc-webgpu-phase-profile-2026-08-29.json`.

- [Probe 2026-08-29] GigaAM RNN-T all-WebGPU session initialization: an
  opt-in `parallelSessionInitialization` flag overlaps creation of the
  independent encoder, decoder, and joint graphs. Three fresh Chrome/NVIDIA
  Blackwell runs reduced median load from `8,821.245 ms` (serial) to
  `7,556.690 ms` (parallel, `14.3353%` lower) while preserving exact 91-token
  parity. The probe is deliberately opt-in: a mixed WebGPU/WASM attempt
  reproduced ORT's `multiple calls to initWasm()` race, and earlier WebGPU
  runs reported concurrent EP-creation failures. Mixed and WASM compositions
  therefore remain serial. Do not promote the flag to a default until another
  browser/adapter and a longer lifecycle soak pass. Evidence:
  `docs/reports/gigaam-rnnt-session-init-concurrency-2026-08-29.json`.

## First inspect the real repository

Before changing anything:

- inspect the current working tree and preserve existing user changes;
- read docs/architecture.md, docs/PROJECT_CHARTER.md, and relevant handoffs;
- inspect actual source, tests, tools, artifacts, and sibling/reference repos;
- distinguish verified implementations from prototypes and artifact-gated work;
- do not infer implementation status from documentation alone.

Keep the existing architecture:

- src/runtime owns orchestration, lifecycle, registration, loading, backend
  selection, and transcript normalization;
- src/models/* owns family-specific preprocessing, tensor wiring, inference,
  decoding, timestamps, and native output shaping;
- src/presets/* owns thin branded configuration and asset resolution;
- src/inference/* contains only genuinely shared descriptors, generic math,
  backend probes, and streaming primitives;
- src/audio/* owns reusable audio and feature-preprocessing primitives;
- src/io/* owns asset providers, external data, and caching;
- src/types/* owns stable contracts.

Do not add framework UI to the core package, copy Transformers.js architecture,
or move model-specific behavior into generic runtime/inference layers without
proven reuse.

## Whole-library product objectives

Continuously inspect and improve the library as a complete developer product.
Choose work from evidence rather than assuming the newest model is always the
highest-value task.

### Public API and developer experience

- Keep the canonical transcript contract stable across models and backends.
- Keep the root API narrow, coherent, typed, and runtime-critical.
- Use intentional subpath exports for builtins, IO, inference, browser,
  realtime, benchmarks, datasets, model families, and presets.
- Make common workflows simple without hiding important model/backend limits.
- Provide actionable errors, progress events, cancellation, capability
  discovery, logging, and deterministic disposal.
- Preserve structured-clone-safe outputs for worker and browser use.
- Improve naming, types, documentation, and examples when users would otherwise
  need to understand internal implementation details.
- Avoid API churn unless the measured benefit justifies migration cost.

### Runtime reliability

- Make repeated model loading, transcription, cancellation, and disposal safe.
- Prevent tensor, GPU-buffer, object-URL, worker, IndexedDB, filesystem-handle,
  and session leaks.
- Keep backend differences from changing canonical transcript semantics.
- Support resilient local and remote asset loading, external ONNX data,
  caching, progress reporting, interrupted downloads, and recovery.
- Keep browser-only code away from Node/root import paths and verify package
  exports, tree-shaking boundaries, and worker compatibility.
- Test silence, malformed input, unsupported capability combinations, repeated
  inference, concurrent activity, and cleanup paths.

### Realtime and long-audio ASR

- Improve microphone capture, resampling, ring buffers, chunking, VAD, rough
  gates, detector/controller behavior, rolling windows, overlap merging,
  partial/final revisions, and retroactive corrections.
- Keep model intelligence inside model families and shared realtime
  orchestration in the appropriate runtime/inference layer.
- Measure end-of-utterance latency, first-partial latency, committed-text
  stability, boundary loss, duplicate text, drift, and long-session memory.
- Do not claim streaming support merely because fixed windows can be looped.
  Document whether a model is truly stateful streaming, chunked offline, or
  short-clip only.

### Examples, demos, and integration surfaces

- Treat examples as maintained products and executable API specifications.
- Keep repository examples small and focused on supported public APIs.
- Keep application-specific UI and framework bindings outside the core package;
  move reusable headless behavior back into the library when justified.
- Provide examples for file transcription, microphone/realtime transcription,
  local and remote models, WebGPU/WASM selection, progress/cancellation,
  timestamps, batch or long-audio behavior where supported, and cleanup.
- Ensure examples do not reimplement preprocessing, decoding, or transcript
  merging that belongs in the library.

### First-class sibling integration projects

The parent workspace contains valuable applications and validation harnesses
outside the `speech-recognition` repository. They are part of the practical
library development system even though they remain separate packages and Git
repositories. Do not ignore them because they are outside the current working
directory.

Inspect the relevant sibling projects under `N:\github\asrjs` before changing
the API or behavior they exercise:

- `benchmark-demo`: model/backend benchmark matrix, dataset-driven runs,
  timing, repeatability, transcript comparison, and JSON/CSV result export;
- `browser-demo`: the focused file-transcription app, hosted and local-folder
  model management, artifact inspection, uploads, dataset samples, API usage,
  backend selection, and transcript/reference comparison;
- `playground`: the developer kitchen-sink for public API design, root versus
  compatibility paths, model management, loading options, progress events,
  canonical/native/raw outputs, and repeat-run diagnostics;
- `streaming-demo`: the primary microphone, realtime, buffering, segmentation,
  VAD, waveform, monitor, and partial/final transcript integration surface;
- `vad-demo`: focused VAD, segmentation, detector, timeline, and boundary
  experimentation;
- `firered-vad-web`: a separate FireRed VAD Node/browser implementation and
  parity/profiling reference from which proven reusable behavior may be brought
  into the core library;
- `webgpu-agent-test`: the exact browser/WebGPU correctness, artifact-matrix,
  parity, memory, and performance harness, especially for the optimized Whisper
  split-graph path.

Preserve and validate important model/backend compositions exposed by these
projects. One key example is Parakeet v3 hybrid execution: the encoder runs on
WebGPU while the decoder runs on WASM. Treat independent encoder and decoder
backend selection as a real library capability and integration requirement,
not as demo-only UI state. Verify the current implementation before changing
defaults because supported combinations may differ by model, precision,
artifact, browser, and ONNX Runtime version.

The responsibility boundary is:

- `speech-recognition` is the source of truth for reusable APIs, model logic,
  backend composition, transcript semantics, lifecycle, and headless browser
  primitives;
- sibling applications are first-class consumers, executable examples,
  integration laboratories, and acceptance surfaces;
- application layout, framework state, and debug-only UI remain in the sibling
  project;
- reusable behavior discovered in a sibling project should be implemented and
  tested in the core library, then consumed by the sibling rather than copied
  independently across apps.

When changing the library, identify affected sibling projects and validate in
proportion to the change:

- public API, model loading, local-folder, caching, or progress changes:
  exercise `browser-demo` and `playground`;
- backend composition, quantization, or Parakeet changes: exercise the hybrid
  WebGPU-encoder/WASM-decoder path and the relevant benchmark/browser project;
- realtime, capture, VAD, chunking, or transcript-merging changes: exercise
  `streaming-demo` and, where relevant, `vad-demo` or `firered-vad-web`;
- WebGPU artifact, cache, precision, or performance changes: exercise
  `webgpu-agent-test` with the exact artifact and warmed browser configuration;
- benchmark API or performance claims: exercise `benchmark-demo` and preserve
  exportable evidence.

A library feature is not complete when its unit tests pass but its relevant
sibling integration project is broken, stale, or forced to reimplement the
feature. Update affected examples in the same implementation cycle when the
public contract intentionally changes. Preserve each sibling repository's
independent Git state and avoid unrelated rewrites.

### Quality, testing, and benchmarking

- Build layered tests: pure unit tests, deterministic fixtures, artifact-gated
  parity tests, Node smoke tests, and independent browser/WebGPU checks.
- Benchmark exact artifact/backend combinations with warmed repeated runs.
- Use a shared cross-model audio window shorter than 30 seconds for comparable
  throughput measurements (the current canonical clip is the 18.714-second
  `librivox.org.wav`; the 11-second JFK fixture remains the labeled smoke
  oracle where a family has one). Run at least one same-session warm-up before
  recording measured repetitions, report warm-up separately, and keep model
  correctness/parity checks distinct from unlabeled throughput runs. Longer
  source recordings such as the 146-second JFK asset may be clipped into
  supported windows, but must not be passed through a graph whose static frame
  limit it exceeds; record the source identity and measured clip duration.
- Track accuracy and output semantics as well as latency, throughput, memory,
  transfers, loading, and cleanup.
- Prefer changes that create measurable correctness, quality, performance,
  usability, or reliability gains. Record negative results and avoid repeatedly
  pursuing low-yield knobs without new evidence.

Model implementation is one major workstream within this broader mission. When
porting a model, follow the mandatory chain below.

## Mandatory original-model reference chain

Do not begin a serious port by only downloading an existing ONNX artifact or
writing a mocked graph boundary. For each selected candidate, establish the
complete reference chain first.

### 1. Obtain the original model assets

Locate and, when artifact access is permitted, download the original model
weights from the official source or the upstream project's recommended source.
This may include PyTorch, safetensors, checkpoint, tokenizer, processor,
configuration, generation, vocabulary, and auxiliary assets.

Record:

- official repository and model identifier;
- exact revision, commit, or release;
- source URLs and license;
- file names, sizes, and hashes where practical;
- framework and dependency versions;
- local cache or external artifact location;
- the exact processor/tokenizer/configuration used.

Do not commit large weights to this repository and do not publish or mutate
external model repositories without explicit approval. If an external or
Hugging Face download requires approval, stop at the access boundary and report
the exact required artifact rather than silently substituting an unrelated
checkpoint.

### 2. Run the official or recommended inference engine

Run the upstream official inference implementation whenever available. If the
project recommends a specific supported engine, use that engine and document
why. A generic Transformers wrapper or an existing ONNX runtime is not a
replacement when it changes preprocessing, decoding, cache behavior, or output
semantics.

Use fixed audio fixtures and capture:

- audio identity, sample rate, channels, and waveform metadata;
- preprocessing inputs and acoustic features;
- encoder inputs and outputs;
- decoder inputs, masks, positions, and KV caches;
- logits and selected token probabilities;
- token IDs, EOS behavior, timestamps, language, and other metadata;
- final transcript and relevant native output.

Save a reproducible reference manifest containing the model revision, weight
source, engine version, command, environment, fixture identifiers, and output
locations. The upstream output is an implementation oracle. Human or benchmark
gold is a separate quality reference; never confuse generated reference text
with ground-truth labels.

Never join evidence by row order or text hash. Preserve stable sample_id or
audio identity and keep labels, model outputs, candidates, scores, and
diagnostics separate.

## Mandatory optimized ONNX artifact chain

The goal is not merely to export a graph that loads. Produce ONNX artifacts
designed for this library's execution paths and measured browser constraints.

### 3. Analyze and design graph boundaries

Study the Parakeet and Whisper ports as examples of complete artifact and
runtime engineering, while preserving this repository's own architecture.
Choose graph boundaries appropriate to the candidate topology, for example:

- encoder plus CTC head;
- encoder, predictor, joiner, and transducer state;
- encoder, decoder-init, and decoder-step graphs;
- prefill and token-step graphs with explicit KV cache;
- alignment or auxiliary graphs only when their contracts are verified.

Do not force every model into Whisper's graph shape. Keep model-specific
decoder logic in its family implementation.

### 4. Export, transform, and audit

Use the existing conversion tools, model-debugging scripts, ONNX exporters,
reference repositories, and sibling-workspace tooling where appropriate.
Improve those tools when the same problem is likely to recur.

For every artifact, inspect and record:

- graph inputs and outputs;
- static and dynamic shapes;
- dtypes and opset;
- operator inventory and unsupported operations;
- external-data files and path contracts;
- parameter size and expected peak memory;
- cache tensor shapes, ownership, and lifetime;
- precision variants and numerical risks;
- WASM, native ORT, and WebGPU provider compatibility.

Create the most suitable artifact variants for the library, such as fp32,
fp16-I/O, pure fp16, int8 or other quantized variants, only when the target
execution provider supports them. Optimize graph boundaries, precision,
external-data layout, output fetches, cache placement, and transfer behavior
for VRAM, CPU, GPU, and WASM constraints.

“Optimized” must be demonstrated by measurements. Do not claim the fastest or
lowest-memory graph from export settings alone.

### 5. Validate the artifact against the original model

Use this validation ladder:

original official engine
    → exported ONNX on native/desktop ORT
    → ONNX on WASM where supported
    → ONNX on WebGPU
    → the library's model-family executor
    → browser end-to-end pipeline

At each stage, compare the earliest meaningful divergence:

audio → features → encoder → decoder state → logits → tokens → text

Report shape, dtype, min/max, mean, standard deviation, NaN/Inf counts,
absolute and relative error, cosine similarity, top-k logits, argmax
agreement, first token divergence, EOS behavior, transcript, and timestamps
where applicable. Use provider-appropriate tolerances and compare the same
artifact and fixture, not unrelated model variants.

Do not promote a WebGPU graph before the native/desktop graph is correct.
Do not treat a mocked session, graph-load success, or one readable transcript
as real model verification.

## Browser and library integration

Browser test pages are validation shells. They must call the library's single
implementation and must not duplicate preprocessing or decoder logic.

Integrate verified work through the real library contracts:

- model family under src/models;
- thin preset only when promotion is justified;
- asset loading, external data, caching, progress, cancellation, and errors;
- canonical transcript mapping and capability metadata;
- correct disposal and repeated-inference cleanup;
- long-audio, streaming, timestamps, and metadata only when actually supported;
- focused CI-safe tests plus artifact-gated parity tests;
- documentation of artifact provenance, license, limits, and measurements.

Then validate the integration through at least one realistic example or sibling
demo appropriate to the feature. Improve the public API or example experience
when integration exposes unnecessary complexity, while preserving the
architecture and stable transcript semantics.

Use native or stable CPU-KV execution as the correctness oracle when
applicable. Keep GPU-KV, batched beam search, speculative decoding, custom
kernels, and similar optimizations opt-in until token/text parity, cache
safety, and memory behavior are independently proven.

## Candidate selection and failure handling

Evaluate FireRedASR2-AED, FireRedASR2-LLM, Qwen3-ASR 0.6B, SenseVoice,
X-ASR, newer streaming models, and other candidates by:

quality × browser feasibility × strategic value

Consider Turkish and multilingual quality, architecture complexity, official
weights and inference availability, exportability, ORT Web support, WebGPU
and WASM viability, model size, VRAM/CPU memory, latency, streaming,
timestamps, quantization, licensing, and engineering effort.

Do not work on every candidate at once. Choose a bounded candidate objective.
If a candidate is fundamentally unsuitable, preserve the evidence, classify
the failure, improve reusable tooling where justified, and move on.

Use explicit failure categories such as:

EXPORT_BLOCKED, ONNX_GRAPH_INVALID, ORT_WEB_UNSUPPORTED_OP,
WEBGPU_UNSUPPORTED_DTYPE, WEBGPU_MEMORY_LIMIT, PREPROCESSING_MISMATCH,
ENCODER_MISMATCH, DECODER_MISMATCH, TOKENIZER_MISMATCH, CACHE_LOGIC_ERROR,
GENERATION_POLICY_ERROR, PERFORMANCE_NOT_VIABLE, MODEL_TOO_LARGE,
LICENSE_BLOCKED, and ARCHITECTURE_NOT_BROWSER_SUITABLE.

Preserve failed experiment details: candidate, revision, original assets,
official engine, export method, command, environment, artifact hashes, result,
diagnosis, and possible next action.

## Performance and promotion requirements

After correctness, measure warmed repeated runs using the exact artifact,
browser, adapter, and backend:

- model load and initialization;
- preprocessing, encoder, decoder, and total latency;
- RTFx and decoder step counts;
- VRAM, CPU, JS, and WASM memory;
- CPU↔GPU transfers and tensor downloads;
- cache storage, copying, and disposal;
- precision and quantization trade-offs.

A candidate is complete only when it has:

- original-weight provenance and a reproducible official/reference run;
- captured reference outputs;
- audited and optimized ONNX artifacts;
- native/desktop parity;
- relevant WASM and WebGPU parity;
- canonical library integration and lifecycle safety;
- representative quality and performance evidence;
- documented limitations, licensing, and artifact boundaries;
- reusable lessons added to tools, tests, fixtures, skills, or documentation.

Every serious port must improve both the supported ASR capability and the
machine used to port future models.

Your first response should inspect the repository and current handoffs, then
identify the highest-value bounded improvement across the API, runtime, model
support, realtime behavior, performance, examples, tests, or tooling.

Explain the user-visible or engineering gain, the evidence that motivates it,
the affected architectural boundaries, the verification plan, and what counts
as completion.

If the selected work is a model port, also show:

1. where the original weights and official inference engine will come from;
2. which reference outputs will be captured;
3. which ONNX graph variants will be produced and why;
4. how native, WASM, and WebGPU parity will be tested;
5. what will count as promotion or closure.
