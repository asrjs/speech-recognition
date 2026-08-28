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
7. reusable model-porting and verification infrastructure.

This is a single-package, ESM-first, speech-focused, headless and
framework-neutral runtime. It is not a generic model zoo or multimodal
framework.

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

