# Whisper WebGPU Completion Plan

Updated: 2026-08-24 (beam selector and browser parity matrix)
Branch: `feat/whisper-cleanup-beam-temperature`

## Direction

Pause broad optimization attempts and finish the practical Whisper runtime path:

1. preserve reference decode semantics,
2. make the quality controls actually work in the runtime wrapper,
3. complete real language detection,
4. keep WebGPU greedy GPU-KV as the measured fast path,
5. make beam search correct first, then optimize it with batched decode later.

`condition_on_previous_text`, hotwords, and numeral suppression are intentionally
deprioritized. Prior model testing showed they are not the highest-leverage work
for this package, and context carryover can amplify mistakes on long audio. Keep
the existing option surfaces, but do not spend the next implementation pass there
unless a reproducible fixture shows a clear win.

## Current State

- `src/models/whisper-seq2seq/executor.ts` owns the model-specific Whisper
  splitgraph bridge, prompt construction, timestamp processing, KV-cache wiring,
  language detection hooks, and transcript assembly.
- `src/models/whisper-seq2seq/core.ts` owns the backend-neutral decode loop for
  greedy, beam, temperature sampling, and best-of retries.
- `src/models/whisper-seq2seq/core.ts` owns the active/finished beam lifecycle,
  candidate expansion, parent-index KV routing, patience budget, and final
  sequence selection. `beam-search.ts` retains lower-level ranking helpers used
  by focused tests and other callers.
- `src/models/whisper-seq2seq/enhanced-executor.ts` wraps the vanilla executor
  with VAD chunking, quality gates, and temperature fallback.
- `src/quality/*` owns model-agnostic quality math and the generic temperature
  fallback loop.
- `src/chunking/*` and `src/post-processing/*` own reusable segment merging,
  drift, and transcript cleanup helpers.

The GPU-KV path is greedy-only by design today. Beam search improves quality, but
it currently belongs on the stable CPU-KV splitgraph path. An opt-in batched beam
path now uses the existing batch-shaped decoder-step inputs and explicit parent
KV routing; it remains experimental until broader fixture and model coverage is
complete.

## Dependency Direction

- Shared quality and chunking modules must not import Whisper model internals.
- `enhanced-executor.ts` may import shared quality/chunking modules and delegate
  all model execution to the vanilla `WhisperExecutor` interface.
- `executor.ts` may import model-specific processors, tokenizer, beam helpers,
  and ORT adapters.
- Runtime/session loading must not absorb model-specific decoder rules.
- Framework adapters remain out of the core package.

## Completed In This Pass

- Beam search decode has focused mock-session coverage via
  `tests/whisper-beam-search-decode.test.ts`.
- Enhanced temperature fallback now passes each retry temperature into vanilla
  transcription instead of repeatedly decoding with the original options.
- The enhanced wrapper preserves any caller-supplied `onTokenLogits` callback
  while collecting logits for quality gates.
- `withTemperatureFallback().attempts` now counts decode attempts, not gate
  evaluations.
- Language-token selection is extracted into a tested helper that scores the
  final decoder logits slice and filters for real Whisper language tokens.
- Beam search now keeps survivor KV caches aligned when completed beams are
  retained alongside active beams during patience-based continuation.
- Splitgraph beam KV caches now preserve fp16 typed-array storage and per-beam
  tensor dims when crossing decoder-init/decoder-step callback boundaries. This
  fixes the browser fp16 beam path without enabling GPU-KV for beam.
- Decode dispatch now follows Whisper/faster-whisper semantics: beam search is
  the `temperature=0` path, nonzero temperature uses sampling, and `bestOf`
  applies only to nonzero-temperature sampling.
- Beam search now follows the OpenAI Whisper candidate lifecycle: EOS sequences
  move to a separate finished set, active slots remain available to non-EOS
  hypotheses, and `round(beamSize * patience)` is the finished-candidate budget.
- Final candidate ranking now uses Whisper's simple length normalization when
  no penalty is specified and the Google NMT formula for an explicit alpha;
  explicit `lengthPenalty: 0` retains raw cumulative-score ranking.
- Survivor KV caches are routed by explicit parent indexes instead of inferred
  token-prefix matching. Stable and opt-in batched execution share this logic.
- Timestamp processing now suppresses `<|notimestamps|>` during timestamped
  decode and applies Whisper's aggregate timestamp-probability rule.
- The reproducibility harness reads graph dimensions from ONNX metadata,
  separates 3000 mel input frames from 1500 encoder output positions, supports
  independent encoder/decoder variant directories, and loads the actual
  `generation_config.json` policy.
- Reference execution honors a single model-directory override ahead of paths
  embedded in generated JSON and converts float32 mel to fp16 when the encoder
  graph declares fp16 input.
- A locally cached `openai/whisper-large-v3-turbo` reference on the JFK fixture
  now matches the fp32 splitgraph path exactly: 31/31 normalized tokens and
  identical text for both exported Python mel and the TypeScript WAV frontend.
- Raw decoder-init logits are now exposed before suppression through the core,
  splitgraph, GPU-KV, and merged-decoder paths. The enhanced quality wrapper
  passes that vector and the model-resolved no-speech token into its gates.
- Selected-beam quality now uses scalar per-token logprob/entropy traces for
  the winning sequence only. Greedy traces are opt-in via `trackQuality`.
  Logprob/entropy gates consume those traces, so the enhanced wrapper no longer
  copies full-vocabulary logits per token.
- Beam expansion now computes log-sum-exp/entropy and bounded top-k candidates
  in one pass without allocating a full-vocabulary log-softmax array per beam.
  Float32 rounding is preserved before ranking so candidate tie behavior stays
  compatible with the previous implementation.
- Added `npm run benchmark:whisper-beam`, a deterministic stable-vs-batched
  contract test for beam sizes 2, 3, and 5. It checks exact token parity and
  decoder-call reduction without asserting machine-dependent wall-clock times.

## Healthy GPU Rerun

After the workstation restart, the existing WebGPU harness again exposed the
NVIDIA Blackwell adapter with `shader-f16`. The active artifact was the custom
`ysdede/whisper-large-v3-turbo-onnx-4graph` splitgraph. The remote encoder
artifact is `fp16_iofp32/encoder_model.onnx`; the local harness uses its
optimized fp16-output copy, `fp16_iofp32_fp16out`, paired with the `fp16`
decoder. A warmed 30-second JFK measurement reached `22.76x` RTFx without GPU
tensor downloads; an independent repeat on the optimized local variant reached
`25.6993x` RTFx (`1175.81ms` total). This confirms the earlier ~8x result was a
degraded GPU state, not a decoder code regression, and validates the historical
`25-28x` range when the correct local variant is selected.

## Priority Plan

### P0: Keep Temperature Fallback Honest

Status: completed for the wrapper-level retry path.

- Verify single-chunk and VAD chunk fallback both pass `temperature` through to
  vanilla decode.
- Keep compression/logprob/entropy/no-speech gates optional and runtime-only.
- Do not let enhanced collection callbacks hide caller diagnostics.

### P1: True Language Auto-Detection

Status: implemented for splitgraph and merged-decoder compatibility paths;
non-English splitgraph browser coverage is validated with the Turkish TDK
fixture (`language=auto` selects `tr`).

Goal: replace silent English/default fallback wherever splitgraph artifacts can
detect language from encoder output.

- Audit `detectLanguageFromEncoder()` against prompt construction and tokenizer
  special-token handling.
- Add tests for language-token selection from mocked decoder logits.
- Add an executor-level test for `language: "auto"` selecting a non-English
  language token from mocked splitgraph decoder outputs.
- Preserve fallback behavior only for merged/non-splitgraph paths that cannot
  run the language probe.
- Surface detection timing and selected language in metrics without changing the
  canonical transcript contract.

### P2: Beam Search Correctness and Quality Parity

Status: stable CPU-KV splitgraph path passes focused unit coverage and current
browser functional validation for fp16 WebGPU. The corrected active/finished
beam lifecycle produced the same 50-token sequence in greedy, stable beam, and
experimental batched beam runs on 2026-08-23.

Goal: make beam search a reliable accuracy option before optimizing it.

- Keep focused coverage for EOS separation, finished-candidate patience,
  length penalty, survivor KV-cache alignment, and timestamp rules.
- Keep `bestOf` as a sampling-only control. Do not combine it with beam search
  as a batched-beam proxy.
- Add an integration-style splitgraph mock that checks token details and final
  text for `numBeams > 1`.
- Keep GPU-KV disabled for beam until KV cloning/reordering is implemented
  without changing output semantics.

Browser validation on 2026-06-19:

- Greedy fp16 WebGPU with GPU-KV: functional pass, zero GPU tensor downloads,
  KV location `gpu-buffer`.
- Beam fp16 WebGPU with `numBeams=2&patience=1` and GPU-KV disabled:
  functional pass on the stable CPU-KV path, decode p50 about 106ms per step,
  KV location `cpu`.
- Experimental batched beam with `numBeams=2&patience=1&batchedBeam=1`:
  functional pass with the same transcript prefix as stable beam. Decoder-step
  ORT calls dropped from `98` to `49`; paired browser measurement improved from
  about `15.16s` total / `1.98` RTFx to about `12.83s` total / `2.33` RTFx.

Browser revalidation on 2026-08-23 after the beam lifecycle correction:

- Greedy GPU-KV, stable beam, and batched beam produced the same 50 generated
  tokens and transcript on the 29.9s JFK fixture. The harness reports `check`
  because `maxNewTokens=50` truncates its longer text oracle.
- Greedy GPU-KV: total `3291.080ms`, decode `2637.220ms`, `9.1304` RTFx,
  zero GPU downloads.
- Stable beam: total `14126.025ms`, decode `12577.285ms`, `2.1192` RTFx,
  98 decoder-step ORT calls.
- Batched beam: total `11841.205ms`, decode `10609.685ms`, `2.5276` RTFx,
  49 decoder-step ORT calls.
- In the paired beam run, batching cut ORT calls by 50%, decode time by 15.64%,
  and transcription time by 16.17%, with exact token parity.

### P3: WebGPU-Safe Beam Optimization

Status: bounded candidate selection is implemented; the experimental opt-in
path is browser-validated for CPU-KV splitgraph beam with the current fp16
WebGPU artifact.

Goal: reduce beam cost after correctness is proven.

- [x] Replace the full-vocabulary log-softmax allocation in beam expansion with
  a fixed-size top-k selector that retains only the candidate set.
- [x] Design and implement a batched decoder-step graph/API that accepts beam-shaped
  `input_ids` and KV tensors.
- [x] Reorder KV by surviving beam parent after candidate selection.
- [x] Compare tokens against the stable beam path before taking timing wins.

Current promotion gate:

- Stable and batched beam must produce identical tokens and text on the same
  artifact and fixture.
- Validate `numBeams=2..5`, early EOS, timestamps, and at least one Turkish
  fixture before making batched beam the default.
- Measure ORT calls and wall time after token parity; a timing win cannot excuse
  a decode-policy difference.

Implemented experiment:

- `experimentalBatchedBeam` transcription option.
- Pure decode loop batches active beams only when the option is true and the
  session exposes `runStepBatch()`.
- Splitgraph bridge builds `[activeBeams, 1]` decoder-step inputs and batched KV
  tensors, then splits logits and present KV back into per-beam caches.
- fp16 KV batching preserves `Float16Array` inputs for browser ORT.
- Batched present-KV outputs are split with zero-copy typed-array views; input
  packing still clones into fresh storage before the next ORT call.
- The current fp16 WebGPU artifact accepts batch-shaped decoder-step inputs, but
  the option remains off by default until wider model/back-end validation is
  complete.

The 2026-08-25 follow-up measured the decoder-step KV merge bucket at `3–5ms`
per batched run after the zero-copy split, versus `12–47ms` before it. Direct
packing then reduced feed-build time over 49 batched calls from about
`2.9–3.1s` to `1.57s` for beam 2 and from `6.94–6.96s` to `3.57–3.63s` for
beam 5. A fresh timestamped 10s browser run retained exact stable/batched
tokens and word timestamps; total wall time remains variable, so the path
stays opt-in.

Browser matrix revalidation on 2026-08-24:

- English 30s, `numBeams=5`: stable and batched tokens matched exactly; stable
  used 245 decoder-step calls and batched used 49. Total time was `24764.285ms`
  versus `18703.1ms`; both used CPU KV and zero GPU downloads.
- English 10s timestamped, `numBeams=2`: stable and batched tokens and all 17
  word timestamps matched exactly; calls dropped from 40 to 20.
- Turkish 18s, `language=auto`, `numBeams=2`: both paths detected `tr`, emitted
  the same transcript, and calls dropped from 158 to 79. GPU-KV remained off for
  both beam paths.

### P4: WhisperX-Style Extras With Clear Boundaries

Goal: keep valuable extras, skip low-yield knobs.

- Keep VAD pre-segmentation using existing TenVAD/FireRed backends.
- Keep word timestamps and DTW alignment in the Whisper model family.
- Use WAV2VEC2 forced alignment only as an optional advanced pass.
- Do not add framework-specific state, UI components, or adapter semantics to
  the core package.

### P5: Quality Metrics From The Correct Decoder Positions

Status: raw no-speech provenance and selected-beam scalar traces implemented;
browser English/Turkish revalidation completed on 2026-08-23 (greedy GPU-KV
`30.4192x` on 29.9s JFK, exact token match vs stable beam 2, Turkish auto-detect
`tr`). Word-timestamp interpolation fallback is in; DTW vs WhisperX remains.

- [x] Capture no-speech probability from the raw decoder-init logits at the SOT
  position, before suppression, using the model's configured no-speech token.
- [x] Keep the compatibility fallback for direct generic gate callers while
  resolving the actual token from generation config or the tokenizer in the
  Whisper executor.
- [x] Thread raw init logits through greedy, beam-init, GPU-KV, and merged
  decoder paths without changing the default fast path when no callback is set.
- [x] Define selected-beam quality metrics without retaining every full-vocabulary
  tensor for every hypothesis.
- [x] Add fixture gates proving compression/logprob rejection and temperature
  recovery after the metric source is correct.

## Assumptions

- WebGPU greedy with `experimentalGpuKvCache` remains the measured browser fast
  path.
- Beam search is a quality feature first and a performance feature only after
  batched decode exists.
- Hotwords and previous-text conditioning stay deprioritized unless a fixture
  demonstrates a measurable improvement.
- Generated model artifacts and large fixture outputs stay out of normal Git
  history unless explicitly requested.
