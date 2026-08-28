# Project Charter

## Mission

Make `@asrjs/speech-recognition` a best-in-class web ASR library: accurate,
fast, memory-efficient, reliable, easy to integrate, and pleasant to use in
browser and local Node.js applications.

Model porting is one important workstream. The project also owns the quality of
its public API, runtime lifecycle, realtime and long-audio behavior, browser
integration, asset system, examples, documentation, tests, and benchmarks.

This charter complements [architecture.md](./architecture.md). The architecture
document defines the code boundaries; this document defines the project-level
priorities and evidence standards.

## In Scope

- A coherent, stable, typed public API and canonical transcript contract.
- Browser and Node.js runtime composition, lifecycle, workers, and backends.
- Asset loading, external data, caching, progress, cancellation, and recovery.
- Realtime capture, resampling, buffering, VAD, chunking, partial/final state,
  transcript merging, and long-audio orchestration.
- Timestamps, alignment, language detection, confidence, quality controls, and
  capability metadata where supported and verified.
- Technical model families under `src/models`.
- Thin branded presets under `src/presets`.
- ONNX artifact inspection and conversion support.
- Reference-versus-port numerical parity.
- WebGPU, WASM, and native ORT validation.
- Audio preprocessing, tokenizers, decoders, caches, timestamps, streaming,
  and lifecycle correctness.
- Reusable model-porting tools, fixtures, diagnostics, and documentation.
- Maintained examples, demos, browser integration harnesses, benchmark tools,
  and developer documentation.

## Out of Scope

- Turning the core package into a generic multimodal framework.
- Copying Transformers.js architecture or public API philosophy.
- Framework-specific UI bindings in the core package.
- Application-specific business logic or UI that belongs in sibling projects.
- Unverified model claims based on mocked graphs.
- Automatic Hugging Face publishing or mutation.
- Large model-weight downloads or commits without explicit approval.
- Premature universal abstractions.

## Current Project Reality

This is a single-package, ESM-first, speech-focused, headless and
framework-neutral runtime. It is not yet a universal model-porting platform.

The core library already owns more than model execution: canonical transcripts,
runtime composition, IO and caching, browser capture, workers, realtime
controllers, VAD and rough-gate behavior, buffering and chunking, waveform and
monitor helpers, benchmark/dataset utilities, and model-family/preset
registration. These surfaces must evolve together as one product.

The repository contains established or developing work for NeMo/Parakeet/Canary
topologies, Whisper seq2seq, LASR/Wav2Vec2 CTC, GigaAM CTC/RNNT, SenseVoice,
Qwen3-ASR, and X-ASR-related paths. FireRedASR2 is documented and has partial
runtime-related work, but must not be described as a completed first-class
FireRed backend until real artifacts, reference parity, and library integration
are verified.

Candidate models are selected by:

`quality × browser feasibility × strategic value`

Important criteria include Turkish and multilingual usefulness, architecture
complexity, ONNX exportability, ORT Web support, WebGPU compatibility, WASM
fallback behavior, model size, memory, latency, streaming support, timestamp
support, quantization potential, licensing, and engineering effort.

## Product Pillars

1. **API:** coherent types, stable transcript semantics, clear capability
   boundaries, useful errors, progress, cancellation, and disposal.
2. **Runtime:** reliable browser/Node execution, workers, assets, caching,
   backend selection, memory safety, and repeated-use behavior.
3. **ASR behavior:** accurate decoding, language handling, timestamps,
   alignment, confidence, realtime, streaming, and long-audio composition.
4. **Models:** high-value verified families and presets with optimized native,
   WASM, and WebGPU artifacts.
5. **Developer experience:** maintained examples, demos, documentation,
   diagnostics, benchmarks, and migration guidance.
6. **Engineering system:** reusable verification, porting, profiling, fixtures,
   failure records, and independent browser validation.

Work should be selected by expected measurable user value across these pillars,
not by model novelty alone.

## First-Class Companion Projects

The sibling projects under `N:\github\asrjs` are part of the library's
development and acceptance system even though they remain separate packages:

- `benchmark-demo` validates dataset-driven model/backend performance and
  repeatability;
- `browser-demo` validates file transcription, hosted and local model
  management, artifact inspection, and common public API usage;
- `playground` validates the broad developer-facing API, loading options,
  progress, outputs, settings, and diagnostics;
- `streaming-demo` validates microphone capture, realtime buffering,
  segmentation, VAD, monitoring, and partial/final transcript behavior;
- `vad-demo` provides a focused VAD and segmentation laboratory;
- `firered-vad-web` is a separate FireRed VAD implementation and
  parity/profiling reference;
- `webgpu-agent-test` validates exact WebGPU artifacts, precision combinations,
  cache behavior, browser parity, memory, and performance.

These projects are first-class consumers and executable examples, not the
architectural source of truth. Reusable logic belongs in
`@asrjs/speech-recognition`; framework state and application-specific UI remain
outside it.

Important compositions implemented by these consumers are library requirements.
This includes Parakeet v3 hybrid execution with a WebGPU encoder and WASM
decoder, plus independent encoder/decoder backend selection where the model and
artifacts support it.

Changes to public APIs or runtime behavior must identify and validate affected
companion projects. A feature is not complete if the relevant companion app is
broken, stale, or must duplicate core preprocessing, decoding, lifecycle, or
transcript logic. Each repository's independent Git state must be preserved.

## Evidence Policy

- A serious port must establish an original-weight provenance chain: official
  or upstream-recommended weights, exact revision, processor/tokenizer/config,
  license, and reproducible local artifact location.
- A serious port must run the official or upstream-recommended inference engine
  where available and capture reference outputs before relying on an ONNX
  artifact. Generic wrapper output is not automatically equivalent.
- The required artifact chain is original engine → optimized ONNX → native ORT
  → WASM/WebGPU where supported → library executor → browser pipeline.
- ONNX artifacts must be designed and measured for the library's VRAM, CPU,
  GPU, and WASM constraints; graph-load success alone is insufficient.
- Reference implementation output is an implementation oracle, not automatically
  human ground truth.
- Benchmark gold, model output, labels, and diagnostics remain separate
  evidence.
- Audio identity must use stable identifiers such as `sample_id`, never text
  hashes or row order.
- Artifact status must be labeled as verified, prototype, or artifact-gated.
- Performance comparisons require the same artifact family, backend, browser
  state, and warmed measurement method.
- Browser test pages must call the library implementation and must not duplicate
  model decoding logic.
- Large original or converted weights must remain outside Git unless explicitly
  approved; record provenance, hashes, and artifact locations instead.

## Porting and Promotion Gates

Every serious candidate should pass the following stages as applicable:

1. Candidate and architecture study.
2. Upstream/reference inference on fixed audio.
3. Intermediate reference capture.
4. ONNX export and artifact audit.
5. Native/desktop ORT validation.
6. Stage-level numerical comparison: audio, features, encoder, decoder state,
   logits, tokens, and text.
7. WASM and WebGPU validation.
8. Canonical transcript, asset lifecycle, disposal, and error handling.
9. Representative quality tests.
10. Performance, memory, and browser measurements.
11. Documentation of limitations, provenance, and reusable lessons.

A model moves from candidate to public preset only after the applicable gates
are satisfied. A mocked graph boundary or one readable transcript is not proof
of an end-to-end model port.

## Optimization Policy

Priorities are:

1. Correctness.
2. Reproducibility.
3. Browser compatibility.
4. Lifecycle and memory safety.
5. Performance.
6. Maintainability.
7. Further optimization.

Native or stable CPU-KV execution is the correctness oracle when applicable.
GPU-KV, batched beam search, speculative decoding, custom kernels, and similar
optimizations remain opt-in until token/text parity, cache safety, and memory
behavior are proven.

## Porting-System Improvement

Each substantial model investigation should produce both:

- progress on the candidate model; and
- a justified improvement to the porting system.

Reusable outputs may include artifact inspectors, compatibility reports,
tensor-differential tools, reference-capture scripts, tokenizer/frontend parity
tests, cache validators, graph transformations, fixtures, failure taxonomy
entries, playbooks, skills, capability metadata, and CI-safe regression tests.

Generalize only after repeated evidence supports reuse. Preserve useful failed
experiments with their candidate, revision, artifact/export method, command,
environment, result, diagnosis, and possible next action.

## Failure Classification

Use explicit categories where useful:

`EXPORT_BLOCKED`, `ONNX_GRAPH_INVALID`, `ORT_WEB_UNSUPPORTED_OP`,
`WEBGPU_UNSUPPORTED_DTYPE`, `WEBGPU_MEMORY_LIMIT`,
`PREPROCESSING_MISMATCH`, `ENCODER_MISMATCH`, `DECODER_MISMATCH`,
`TOKENIZER_MISMATCH`, `CACHE_LOGIC_ERROR`, `GENERATION_POLICY_ERROR`,
`PERFORMANCE_NOT_VIABLE`, `MODEL_TOO_LARGE`, `LICENSE_BLOCKED`, and
`ARCHITECTURE_NOT_BROWSER_SUITABLE`.

If a candidate is fundamentally unsuitable, close it cleanly with evidence,
capture useful lessons, and move to the next candidate rather than remaining
indefinitely blocked.

## Success Metrics

Track where practical:

- time from candidate selection to valid artifact;
- time to locate first numerical divergence;
- number of manual graph edits;
- number of reusable tools involved;
- number of regression fixtures;
- model-specific versus shared code;
- WebGPU/WASM compatibility;
- end-to-end latency and RTFx;
- memory and tensor-transfer behavior;
- candidates closed with useful failure evidence.
- public API complexity and common-workflow setup cost;
- package import and browser-worker compatibility;
- model load reliability, cancellation, and repeated disposal behavior;
- first-partial and end-of-utterance latency;
- long-session memory stability and transcript revision stability;
- example coverage of supported public workflows;
- documentation and example parity with the current API.
- affected companion-project build, integration, and browser validation;
- verified hybrid and split-backend execution paths where supported.

## Definition of Success

The project succeeds when users can integrate dependable, fast, high-quality
ASR into web applications through a coherent API and verified examples, while
the repository continually improves its models, runtime, realtime behavior,
performance, tests, diagnostics, and porting system.

Each major change should produce a measurable improvement in at least one
product pillar without silently regressing the others. Each serious model port
should still produce both a verified ASR capability and a better workflow for
porting the next model.
