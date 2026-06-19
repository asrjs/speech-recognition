# Whisper WebGPU Completion Plan

Updated: 2026-06-19
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
- `src/models/whisper-seq2seq/beam-search.ts` owns beam ranking and length
  penalty helpers.
- `src/models/whisper-seq2seq/enhanced-executor.ts` wraps the vanilla executor
  with VAD chunking, quality gates, and temperature fallback.
- `src/quality/*` owns model-agnostic quality math and the generic temperature
  fallback loop.
- `src/chunking/*` and `src/post-processing/*` own reusable segment merging,
  drift, and transcript cleanup helpers.

The GPU-KV path is greedy-only by design today. Beam search improves quality, but
it currently belongs on the stable splitgraph path because each active beam still
requires a separate decoder step. Batched beam decode should wait for an explicit
graph/runtime change that supports beam-shaped decoder inputs and KV reordering.

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

## Priority Plan

### P0: Keep Temperature Fallback Honest

Status: completed for the wrapper-level retry path.

- Verify single-chunk and VAD chunk fallback both pass `temperature` through to
  vanilla decode.
- Keep compression/logprob/entropy/no-speech gates optional and runtime-only.
- Do not let enhanced collection callbacks hide caller diagnostics.

### P1: True Language Auto-Detection

Status: helper and private executor probe covered; public transcript-path
coverage still pending.

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

Status: stable CPU-KV splitgraph path passes focused unit coverage and browser
functional validation for fp16 WebGPU.

Goal: make beam search a reliable accuracy option before optimizing it.

- Extend beam decode tests for EOS, length penalty, patience, survivor KV-cache
  alignment, and timestamp processor interaction.
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

### P3: WebGPU-Safe Beam Optimization

Goal: reduce beam cost after correctness is proven.

- First replace any remaining full-vocabulary sort/allocation hot spots with
  fixed-size selection helpers.
- Then design a batched decoder-step graph/API that accepts beam-shaped
  `input_ids` and KV tensors.
- Reorder KV by surviving beam parent after candidate selection.
- Compare tokens against the stable beam path before taking timing wins.

### P4: WhisperX-Style Extras With Clear Boundaries

Goal: keep valuable extras, skip low-yield knobs.

- Keep VAD pre-segmentation using existing TenVAD/FireRed backends.
- Keep word timestamps and DTW alignment in the Whisper model family.
- Use WAV2VEC2 forced alignment only as an optional advanced pass.
- Do not add framework-specific state, UI components, or adapter semantics to
  the core package.

## Assumptions

- WebGPU greedy with `experimentalGpuKvCache` remains the measured browser fast
  path.
- Beam search is a quality feature first and a performance feature only after
  batched decode exists.
- Hotwords and previous-text conditioning stay deprioritized unless a fixture
  demonstrates a measurable improvement.
- Generated model artifacts and large fixture outputs stay out of normal Git
  history unless explicitly requested.
