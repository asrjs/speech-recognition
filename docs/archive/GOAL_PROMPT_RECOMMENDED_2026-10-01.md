# Goal: Build a compact, capable multi-model speech runtime for web, Node.js and Bun

Updated: 2026-10-01. This is the recommended operational goal prompt. Older
checkpoints in `GOAL_PROMPT.md` are evidence and history, not a competing work
queue. Read this prompt together with `PROJECT_CHARTER.md`, `architecture.md`,
`AGENT_SAFETY_RULES.md`, and the current task list before implementation.

## How to use this prompt

The copy-ready block below is the detailed goal to give a repository-aware
agent. The remaining sections are its operational reference, not a second goal
or task queue. Alternatively, ask the agent to read and follow this entire file.
You do not need to paste the charter as well when the agent has repository
access: the charter, architecture and safety rules remain required companion
documents. This document defines future development work; preparing it does not
itself authorize starting that work.

For a later session with repository access, give the agent this file as its
detailed goal reference. If you prefer a self-contained
message, copy the entire first `GOAL:` block, including its ten sections. The
charter need not be pasted separately; the prompt requires the agent to read it.
Do not copy the dated checkpoint as a permanent statement of model support.

This version consolidates the initial end-to-end model-porting request with the
later whole-product intent: a generic, compact, capable and easy-to-use speech
library, separately verified native Node/Bun execution, first-class companion
examples, and a progressively better engineering system. The older attached
porting proposal is background, not a mandate to build a universal decoder,
invent new folder layers or let research replace useful product delivery.

The copy-ready goal contains durable requirements. Model progress belongs in
the dated checkpoint and task receipts, which must be refreshed before resuming.
Creating or revising this prompt is documentation work, not an instruction to
activate a persistent goal, run inference or resume implementation now.

When ready to start or resume, this is enough to give a repository-aware agent:

```text
GOAL: Build a compact, capable, easy-to-use multi-model speech runtime for
browsers and native Node.js/Bun, while making each subsequent model port
safer, more reproducible and less labor-intensive.

Treat best-in-class as a measurable product ambition, not a claim of global
optimality. Deliver finite, independently verifiable improvements in quality,
usability, latency, memory and reliability. Adding more model names, more
scripts or more abstractions is not itself progress toward this outcome.

Use docs/GOAL_PROMPT_RECOMMENDED.md as the detailed operational reference
for this goal. Improve the whole @asrjs/speech-recognition product: public
API, runtime, supported speech workflows, models, examples, documentation
and engineering tools. Model porting is an important lane, not the entire
project. Our distinctive work is doing browser/WebGPU ports ourselves;
the absence of a ready-made browser implementation is not a rejection
criterion. I own parakeet.js; build on our verified Parakeet v2/v3 and
Whisper experience without copying Transformers.js architecture or API.
We are building a usable speech library first, not merely a collection of
model converters, benchmarks or diagnostic scripts. Preserve and improve
existing valuable workflows while adding new capabilities; judge model and
tooling work by the verified product improvements it enables.

1. Reconcile the actual project before choosing work.
Read AGENTS.md, docs/AGENT_SAFETY_RULES.md, docs/architecture.md,
docs/PROJECT_CHARTER.md, docs/WORKSPACE_CONTEXT.md,
docs/MODEL_HARNESS_CONTRACT.md and docs/PARAKEET_VARIANT_TASKS.md, together
with the current whole-product tasks. Inspect Git/worktrees, inherited
changes, public exports, family implementations, local artifacts, receipts
and companion apps.
Treat old prompts, research lists and chat claims as context, not current
proof or authority. Reuse verified assets and completed work. Distinguish
implemented, experimental, artifact-gated and genuinely verified support.

2. Preserve the real architecture while improving shared infrastructure.
Keep one speech-first, ESM-first, headless, framework-neutral TypeScript
package with a coherent root API and intentional lightweight subpaths.
Use actual source boundaries: types for contracts; runtime for orchestration
and lifecycle; io for assets; audio and existing processors for reusable
audio preparation; inference for proven shared descriptors/primitives;
models/<technical-family> for family math, tensor wiring, decoding and
state; presets/<brand> for thin configuration and weight locators. Preserve
the existing pipeline, chunking, CTC, alignment, quality and post-processing
contracts. Do not invent removed folder layers or a universal decoder.
Share neutral tooling and runtime contracts, but keep genuinely model-
specific implementations and tests separate. A fix for one model must not
silently alter another family's behavior. Keep framework bindings outside
core and canonical transcripts stable and structured-clone-safe.

3. Deliver the whole product, not only new models.
Make common workflows simple by default, with explicit typed composition
for advanced users. Convenience APIs and expert-level APIs must use the same
validated runtime and family implementations, not divergent hidden pipelines.
Improve typed loading, family/preset selection, capability discovery,
local/remote model management, progress, cancellation, reset, disposal and
useful errors. Make PCM/file transcription, microphone/realtime usage,
long audio and supported batching straightforward through public APIs.
Keep minimal common-workflow examples and advanced composition consistent:
users should not need private imports, hidden environment flags, copied model
loops or knowledge of another family's quirks to use a supported feature.
Keep model-independent contracts configurable and typed; unsupported model
capabilities must return an actionable error or explicit unsupported status.
Improve VAD, resampling, buffers, segmentation, overlap merging, partial/
committed/final text and supported timestamps/alignment/diarization.
Do not fabricate capabilities or call windowed offline ASR native streaming.
Validate repeated reuse, interrupted loads, model switching, stale callbacks,
workers, concurrency and bounded memory. Share behavior across models and
hosts without requiring applications to maintain their own decoder.
Make ownership safe before and during asynchronous initialization: roll back
failed acquisitions, reject creation after disposal, test initialization/
disposal races and await ongoing cleanup on repeated disposal calls. Verify
actual partial engine-resource release separately from mocked wrapper tests.

4. Make execution genuinely portable and compact.
WebGPU is the primary browser backend; WASM is an intentional fallback or
per-component companion. Preserve independent encoder/head placement,
including Parakeet v3 WebGPU encoder plus WASM decoder/joint. Test actual
native ORT inference separately in Node and Bun; browser WASM in Node,
Python ORT success or an import-only test is not native library acceptance.
Record actual host, engine, provider and precision per graph/component.
Reject unsupported explicit requests rather than silently substituting a
stub or another engine. Keep native bindings out of browser imports and
DOM-only behavior out of native paths. Measure published/install size,
minimal browser bundles, workers, startup and memory; lazy imports do not
remove the installation cost of regular engine dependencies.

Validate consumer-facing packaging as well as the source checkout. Check the
actual npm payload, export/declaration pairs and a minimal installed-consumer
path when changing exports or dependencies; sibling source aliases alone
cannot prove package usability. Native support claims must name tested Node/
Bun versions and actual providers, not imply every host or model is supported.

5. Follow the confirmed model order with bounded feasibility decisions.
Continue Orukeet first through the existing NeMo TDT/Parakeet v3 family.
Then audit and port genuine smaller Parakeet student/distilled checkpoints
and useful derivatives; fewer file bytes from fp16/int8 packaging are not
fewer parameters. Orukeet is a roughly 627M-parameter fine-tune, not a small
distill. Next rank high-value non-NVIDIA ASR and related speech candidates,
including our original FireRed and Qwen ASR 0.6B interests and the newer
research leads. Verify exact upstream identities, licenses, architecture,
Turkish/multilingual value, quality, memory, latency and conversion routes.
Use the dated Cohere, Voxtral, Moonshine, Fun-ASR, ARK-ASR, Phonon, GigaAM,
alignment and diarization notes as research leads, not verified availability
or permission to bypass the confirmed order. Check current official sources.
No new NVIDIA model work beyond existing Parakeet TDT v2/v3 for now.
Nemotron 3.5 Streaming remains paused; preserve its evidence without letting
it monopolize the queue. Defer unsuitable candidates with a bounded,
evidence-backed explanation, not an unsupported claim that the model is bad.

6. Take every serious port through the original-weight evidence chain.
Acquire or reuse verified original official/recommended PyTorch/NeMo/etc.
weights, not just somebody else's ONNX/GGUF. Reuse verified cached originals;
obtain explicit approval before new large weight downloads, as the charter
requires. Pin revision, license, config, tokenizer and hashes. Run the
official/recommended original inference engine
and capture an implementation oracle on identified representative audio.
Bind captures to the actual effective float32 PCM, rate, channels, valid
length, padding and resampling contract: the same WAV hash is insufficient.
Keep human ground-truth labels separate from official model output.

Export our own family-appropriate fp32 ONNX graphs using the existing tools
and our Parakeet/Whisper optimization lessons. Preserve dynamic lengths,
dtypes, layouts, state/cache contracts, token IDs and external-data integrity.
Compare numerical stages from audio/features through encoder, decoder/state,
logits, decisions, full tokens, timestamps and final text where applicable.
Use explicit tolerances and max/mean errors, allclose/cosine, mismatch counts
and the first divergence. Shape agreement or readable text is not parity.
Keep recorded-state replay distinct from independently accumulated decoding.
Fix the earliest divergence without weakening acceptance or hardcoding the
fixture. Validate the actual library in native Node, native Bun, WASM and
real browser WebGPU/hybrid separately for the configurations being promoted.

Only after fp32 acceptance, optimize graph boundaries, CPU/GPU transfers,
VRAM/RSS/WASM memory, caches, loading and warm latency using matched A/B
controls. Test fp16 against the same oracle before accepting it. Do not
default to int4/int8; that needs a separate justified size/quality/performance
decision. C++/sherpa/community export recipes inform our work but do not
replace original-engine validation. Promote models only with representative
held-out quality, public API, lifecycle, examples and documented limits.

7. Treat companion examples as part of product acceptance.
Inspect the parent asrjs workspace, including benchmark-demo, browser-demo,
playground, streaming-demo, vad-demo, firered-vad-web, webgpu-agent-test and
parakeet-ui-webgpu. Check each app's actual dependency/alias wiring and API
usage. They exercise model management, streaming, VAD, diagnostics and hybrid
execution. A separate FireRed VAD app does not prove integrated FireRed ASR.
Keep relevant examples executable and synchronized with public APIs; move
proven reusable headless behavior into core, leaving app/framework UI outside.
Identify affected consumers and validate their relevant builds/smokes/browser
loops. Respect independent Git states and sibling write permissions.

8. Improve the harness and tools while delivering useful increments.
Reuse and strengthen our reference capture, own exporters, stage comparator,
fixture smokes, simulations and browser WebGPU loops, as we did for Whisper.
Share configurable artifact/audio selection, identity checks, metrics,
timeouts, receipts and cleanup; family adapters own their graph/decode rules.
Remove discovered hardcodes, false PASS cases and unsafe fallback behavior
with focused regressions. An unrequested optional test genuinely SKIPs;
requested missing/corrupt inputs FAIL. Simulations do not prove numerical
parity or real GPU execution. Browser harnesses call the library, not a second
decoder. Retain full output comparisons and await cleanup before success.
Each substantial investigation should leave a useful tested recipe, tool,
regression, fixture or failure record, without manufacturing abstractions.
Classify failures by their actual boundary: acquisition/export, artifact,
environment/harness, operator/provider, input preparation, numerical stage,
decode/cache, lifecycle, quality or performance. Do not hide a harness defect
with model-specific production branches. Keep requested failures visible and
bind receipts to the code/build and artifacts that really executed; an old
PASS does not certify a later revision. Keep lightweight CI checks distinct
from opt-in real-weight, native-engine and browser/GPU acceptance runs.

9. Work from a persistent, evidence-backed plan.
Maintain the existing G0-G6 whole-product tracks and P01-P10 model lane;
do not create a competing queue. Choose bounded outcomes with dependencies,
owned files, affected user journeys, baseline, acceptance checks, commands,
evidence and next steps. Let independent API/runtime/example work progress
alongside model dependencies. Refresh dated receipts after affected edits.
Read the dated Orukeet checkpoint and reconcile each required gate with live
code and receipts before deciding the next experiment. Recorded output-only
controls do not establish feature numerics, source-audio preparation, broader
quality or browser/fp16 acceptance. Do not restart completed downloads/exports
or apply another unproven processor replacement.
Use the latest source-bound delivery handoff as a resume aid, not as permanent
acceptance of later code. Reuse completed public examples and their adapters;
extend the first missing consumer workflow rather than writing a second runner.

For each task persist: ID/track, user-visible outcome, dependencies, owned
files and affected examples, baseline, required acceptance checks, commands,
evidence paths, status, measured change, unresolved gates and next smallest
action. Keep this record proportionate to the work; do not turn paperwork
into a prerequisite for a small useful API or example improvement.

10. Define finite delivery contracts within the whole-product mission.
Translate the current backlog into a finite initial delivery milestone set,
recorded in the existing task list. Name required user journeys, supported
model/artifact/host/provider/precision combinations, acceptance checks and the
evidence that would close each milestone. Do not silently drop required gates,
reorder models or turn a future research list into an infinite completion test.
Keep the long-term mission distinct from the current delivery finish line; if
the whole goal has no agreed finite finish line, ask before declaring it complete.

Use the first delivery cycle to produce a usable increment, not an endless
research program: verified Orukeet fp32 inference through the original-engine
chain and the requested real native/browser configurations; executable public-
API examples for browser file/realtime and local native Node/Bun use; measured
compactness, loading and lifecycle improvements with cross-family regressions;
and a strengthened harness plus an audited next-model backlog. Define the
exact fixtures, host/provider matrix and measured acceptance budgets from the
live baseline before implementation. Future research candidates are not all
mandatory ports for this delivery cycle, but their deferral must not erase
required current gates or change the confirmed model order.

Keep an explicit acceptance matrix for the selected user journeys and model
configurations. Model rows identify family/candidate, artifact revision/hashes,
host/version, actual engine/provider per component, processor, precision,
fixture and oracle. Supply reproducible public-API commands and preserve failed
or untested rows. Do not assume every graph supports every engine; declare
supported combinations and justify genuinely inapplicable checks. Shared fixes
need neutral regressions plus affected-family controls; family-specific math
and expected decoding behavior remain in separate implementations and tests.

For each slice state the user-visible outcome, hypothesis or implementation
change, affected public workflows and acceptance gates. For substantial
numerical or feasibility investigations also record a proportionate time/
compute/storage budget; routine API/example/docs edits need no budget ceremony.
Prefer the smallest experiment that distinguishes
the competing causes; do not rerun an unchanged expensive pipeline without
new evidence. When an investigation reaches its limit, preserve its findings
and continue independent agreed product work. Ask before materially changing
the confirmed model queue or taking an action that needs new authority.

A milestone is delivered when its relevant implementation, public-API use,
regressions, affected examples and documentation pass the stated checks, with
unsupported combinations still explicit. Model milestones additionally need
the relevant original-engine, numerical, real backend and quality evidence.
Leave a runnable example or recipe and a precise resume point. Better tools
must help ship the library; more scripts, a passing simulation, an export or
one matching transcript are not substitutes for a usable verified increment.
The delivery handoff must show which configurations and workflows are verified,
experimental, unsupported or untested, and link their evidence. Report library
footprint separately from weight size and inference memory, and original-engine
parity separately from held-out human-label quality. Explicitly deferred work
is not a passing gate; changing a required outcome needs the user's direction.

On activation, first reconcile G0 and record a short project map plus the
next dependency-ordered slices in the existing backlog. Then execute the first
safely actionable slice, validate its public workflow and update the record.
Do not stop at creating another plan when implementation is authorized, and do
not begin implementation merely because this goal prompt was drafted.

Use surgical read -> minimal edit -> diff -> focused validation -> measured
regression checks. Preserve inherited changes and immutable source/oracle/gold
data. Do not weaken tolerances, erase failed evidence, publish/upload/commit,
install heavyweight toolchains or change external settings without authority.
Delegate only independent, safely isolated work with clear ownership.

Report PASS, FAIL, PARTIAL, UNTESTED and SKIP honestly, with concrete evidence,
measured improvements, unresolved gates and the next actionable step. Define
finite delivery milestones within this ongoing mission; never declare the
entire library finished because one model or one test lane passes. Continue
safely actionable agreed work, and ask when a material scope or authority
decision is needed. The outcome is a better usable speech library and examples,
verified valuable model support, and a better system for the next port.
```

The goal's outcome, verification and constraints follow the structure described
in [OpenAI's Goals guidance](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex).
Project-specific requirements come from the user's decisions and this checkout.

Repository paths and dated checkpoints below describe this checkout, not
hardcoded runtime defaults. Resolve the actual workspace and refresh evidence
when resuming elsewhere. Attached research and earlier agent replies are
background material, not authority to override the user's latest scope or the
repository's current instructions.

Keep two layers distinct: the mission, architecture and product tracks are the
durable goal; the dated checkpoint is a resumable snapshot. Refresh task status
from live code and receipts before acting. Use `PROJECT_CHARTER.md` for product
principles, `architecture.md` for boundaries, `PARAKEET_VARIANT_TASKS.md` for the
ordered model lane, and the existing product task records for API/runtime/example
work. Do not create another competing goal or duplicate implementation queue.

## Mission

Continue developing `@asrjs/speech-recognition` in
`N:\github\asrjs\speech-recognition` into a best-in-class, compact,
multi-model ASR and speech runtime: accurate, fast, memory-efficient,
reliable, capable, and genuinely easy to use in browsers and local Node.js/Bun
applications. Native inference in Node/Bun is a product goal, not a synonym for
running browser WASM in Node. Verify each environment and backend separately.

Improve the whole library, public API, runtime, examples, documentation and
engineering system. Model porting is an important workstream, not the entire
mission. Our distinctive work includes doing browser/WebGPU ports and
optimizations ourselves. A candidate lacking an existing browser package can
be a valuable porting opportunity; it is not automatically unsuitable.

Implement and validate useful increments, not just research summaries or plans.
Maintain an ordered persistent backlog and continue from the first unfinished
task. Balance active model work with high-value product improvements; neither
model novelty nor an endless generic refactor should consume the entire project.

Produce three coupled outcomes: a better usable speech library and examples,
more verified high-value model support, and a more reusable engineering system.
Do not substitute a pile of research scripts for a usable library, or a broad
infrastructure rewrite for a working, measured product increment.

### Meaning of generic compact and capable

- **Generic:** several technical families use coherent loading, assets,
  capabilities, lifecycle and transcript contracts. It does not mean one
  universal processor, decoder, tokenizer or graph contract.
- **Compact:** consumers pay for the families, engines and features they use.
  Measure the published package, minimal browser imports, workers and optional
  native dependencies separately. Library footprint, parameter count, weight
  file size and peak inference memory are different quantities.
- **Capable:** support practical speech workflows, including realtime,
  long audio, VAD, timestamps, alignment and diarization where justified.
  Expose verified capabilities and composition rather than pretending every
  model provides every feature.
- **Easy to use:** common workflows work through documented public APIs with
  useful defaults; advanced users can still control sources, precision and
  component placement without maintaining another decoder in their app.
- **Browser-first and portable:** doing the browser port is our work. WebGPU
  encoder plus WASM decoder/joint is an intentional product composition with
  independently chosen component placement. Native Node and native Bun are separately tested
  execution lanes, not implied by browser or Python success.

## Start by reconciling the real project

Before choosing implementation work:

1. Read repository instructions, `AGENT_SAFETY_RULES.md`, `architecture.md`,
   `PROJECT_CHARTER.md`, `MODEL_HARNESS_CONTRACT.md` and the current task list.
   Inspect `package.json`, public exports and actual implementations. Older
   prompts, attachments and research notes provide context, not a replacement
   architecture or permission to ignore newer user decisions.
2. Check the active checkout, branch, worktrees and dirty changes. Identify
   which changes belong to earlier work and preserve them. Inspect relevant
   handoffs/history when available; do not infer completion from chat claims.
3. Inventory existing model families, presets, asset/runtime helpers, export
   tools, local model-debugging playbooks, reference environments, fixtures,
   simulations, smoke tests and companion applications. Reuse verified local
   weights and tools instead of downloading or rebuilding them unnecessarily.
   Reconcile the family inventory against actual source folders, registration
   and package exports: design-document examples are not a complete support
   matrix. Classify each path as established, experimental, artifact-gated or
   incomplete, with evidence rather than inferring support from a folder name.
4. Reconcile each task with its actual evidence: implementation present,
   experimental, numerically verified, application-verified or still untested.
   Select the first actionable unfinished priority and name its acceptance
   checks before editing. Existing partial work is a starting point, not a
   reason to restart the project or announce an already completed port.

## Preserve the actual architecture

This is one speech-first, ESM-first, headless, framework-neutral TypeScript
package, with one release flow, a coherent root API and intentional subpath
exports. It is not a Transformers.js clone, model-zoo framework or collection
of internal workspace packages. Learn from the user's `ysdede/parakeet.js`,
Whisper ports and `ysdede/transformers.js` without copying the latter's
architecture, folder hierarchy or API philosophy.

Keep responsibilities in the existing source boundaries:

- `src/types`: stable contracts, capabilities and metric definitions.
- `src/runtime`: composition, registration, orchestration, lifecycle and
  canonical transcript handling, not model-specific mathematics.
- `src/io`: asset providers, source resolution, caching and external data.
- `src/audio` and existing processors: reusable PCM, resampling, framing and
  feature extraction; inspect current folders before choosing a location.
- Existing `src/ctc`, `src/chunking`, `src/post-processing`, `src/quality`,
  `src/alignment` and `src/pipeline`: preserve their actual reusable contracts
  and dependency boundaries. Do not assume all speech composition already lives
  in `audio`, or invent a new processors folder merely to match an older prompt.
- `src/inference`: genuinely shared descriptors, backend probes, math and
  streaming primitives.
- `src/tokenizers`: reusable tokenizer contracts and implementations.
- `src/models/<technical-family>`: family-specific feature glue, tensor
  contracts, inference, decoding, recurrent/KV state and timestamps.
- `src/presets/<brand>`: thin configurations, metadata and weight locators,
  never another inference pipeline.

Use the current code and `architecture.md` to resolve older notes. Do not
revive removed encoder/decoder/decoding folder layers or add generic utility
blobs. Shared contracts sit upstream of family implementations; presets and
apps consume them, not the reverse. Reuse a family for compatible weight
variants instead of copying its executor. Share only what is actually shared.

The core can own headless capture, resampling, buffers, chunking, VAD, detectors,
controllers, waveform/canvas renderers and monitor snapshots. React/Vue/Svelte/
Solid bindings belong in thin sibling adapters; framework state and components
must not enter the core. Keep canonical transcripts structured-clone-safe and
semantically stable across models, backends and environments.

Keep the root's coherent runtime, loading and canonical-result API, but preserve
the documented subpaths for builtins, IO (including opt-in `io/node`), inference,
browser/media, realtime, benchmarks, model families and presets. One package and
one coherent API do not mean forcing every feature and engine into the minimal
root import graph. Use "Feature Extractors" or "Processors", not "Frontend",
for audio preparation.

## Whole-product workstreams

### Simple, capable public API

Make common workflows straightforward: select/load a model, transcribe PCM or
supported audio sources through the appropriate decoding helpers, run
microphone/realtime transcription, inspect timestamps and outputs,
change supported backend options, cancel, reset and dispose. Keep typed options,
useful defaults, capability discovery, progress, actionable errors and clear
resource ownership. Advanced control should remain possible without making
every user understand ONNX internals. Do not invent timestamps, confidence,
languages, diarization or streaming capabilities a model cannot supply.

Preserve coherent direct-family and branded-preset loading. A generic runner
must not select Parakeet configuration for every unknown model. Missing real
artifacts must never silently invoke a stub and be reported as real inference.
API changes need migration guidance and affected-example validation.

Build on the existing `createSpeechRuntime`, `loadSpeechModel`, one-shot
transcription helpers and `createSpeechPipeline` surfaces. Inspect their current
types and implementation before choosing a change; do not add a competing
application-level loader or cached inference pipeline merely for a new model.

Use these concrete user journeys as API acceptance targets, adapting to the
actual supported API rather than inventing function names in documentation:

- A minimal PCM transcription example can select two verified families through
  the same runtime/session contracts and consume the same canonical result
  shape. Model-native diagnostics remain optional and clearly separated.
- A capability-gated batch example exercises the existing optional
  `transcribeBatch` contract with mixed-length inputs, preserved result order,
  canonical/native response flavors and explicit empty-input behavior. Check
  `supportsBatch`; unsupported sessions must not pretend serial inference is
  graph batching. Preserve the existing distinction between batch inference and
  long-audio windowing, and validate batch padding/masking against single-input
  controls for the families that actually support it.
- Browser examples cover decoded files and microphone sessions, model download
  and local selection, progress, cancellation and clean disposal. Realtime
  examples make committed/preview/final text and reset behavior understandable.
- Local Node and Bun examples load real local artifacts and use explicit native
  inference where verified, without DOM polyfills or browser-only asset APIs.
  Report unsupported families/providers precisely instead of silently using
  WASM or a stub while presenting the result as native.
  Provide a small runnable consumer example through the public runtime and
  session API, not only a model-debugging inspector. Show local asset selection,
  canonical results, progress/errors and awaited cleanup, with independently
  exercised Node and Bun commands and their actual engine/provider versions.
- Invalid sources, unsupported capabilities, interrupted loads and partial
  initialization have useful typed errors and tested cleanup. A model switch
  does not retain the previous model's buffers, workers or decoder state.

### Capability-driven multi-model composition

Use the existing family/preset registration and capability contracts to expose
what each model can actually do. Keep model identity, technical architecture,
artifact variant, environment, ORT engine, component provider and precision
distinct. Share verified metadata with loading, diagnostics, examples and
benchmarks instead of reproducing contradictory model lists in each consumer.

Record language coverage, input requirements, true streaming versus windowed
operation, timestamps/alignment/diarization, relevant audio limits and tested
backend combinations. Distinguish candidate, experimental and verified status;
capabilities may depend on the artifact and environment, not only the brand.
An import test, installed native dependency or successful capability probe is
not proof of model inference. Reject unsupported explicit requests clearly.

### Compact, portable runtime

Keep browser-only APIs off Node/Bun paths and Node/native bindings off the
browser root import graph. Preserve narrow exports, tree-shaking and explicit
composition; load heavy engines, optional families and tools only when needed.
Measure package/install size, minimal import closure, browser bundle size and
startup cost rather than claiming compactness from file counts.

The current root re-exports backend factory modules, including `native-cpu`.
A browser-safe factory export is not the same as importing native ORT bindings.
Inspect the actual static import closure and exercise the existing root-import
guard before changing exports; neither a subpath name nor a lazy-looking API
alone proves environment isolation.

Include install-time engine cost in compactness acceptance. The current package
declares both `onnxruntime-node` and `onnxruntime-web` as regular dependencies;
lazy imports alone do not remove their installation/download cost. Establish
browser-only and native-consumer baselines, then evaluate a compatible optional
engine installation/loading design within the single-package architecture.
Do not prescribe dependency changes before checking package consumers, engine
availability, lockfiles, published exports and clean-install behavior. Report
what improved and what cost remains; do not claim unused engines are free.

Keep browser WebGPU primary, with WASM fallback and independently configurable
component placement when supported: for example Parakeet v3 WebGPU encoder +
WASM decoder/joint. For Node/Bun, design and verify explicit native ORT execution
and supported native providers without requiring browser polyfills. Reuse
environment/asset/backend boundaries; do not create a separate public API or
copy an ASR executor for each host. Report the actual engine/provider used.
Distinguish desired support from implemented and tested support.

Treat ORT Web versus native ORT as engine selection, and CPU/GPU or per-graph
placement as a related but separate concern. Inspect existing contracts before
choosing the implementation seam. Generic engine loading and environment checks
must not contain model-ID branches; family executors still own their sessions,
tensors and decoding lifecycle. Preserve browser-safe imports and existing
hybrid behavior. Node's current declared minimum is 22; document the actual
Node and Bun versions used for native acceptance rather than assuming Bun
works because Node does.

Improve local/remote model management, pinned revisions, large external-data
loading, caches, download progress, interrupted-load recovery and cancellation.
Avoid unnecessary complete-file copies and tensor readbacks. Validate repeated
load/transcribe/cancel/reset/dispose, partial initialization failure, worker
termination, concurrency limits and bounded memory. Do not silently change
process-wide settings or weaken TLS to make a download work.

### Speech behavior beyond short clips

Improve resampling and feature extraction, VAD/rough gates, capture, ring
buffers, segmentation, chunk scheduling, overlap merging, committed/preview
text and finalization. Test silence, speech/silence transitions, uneven chunks,
long recordings and repeated sessions. Distinguish true stateful streaming
from windowed offline transcription; do not market a loop as native streaming.

Exercise the real `StreamingTranscriber` contract: `pushAudio`, `flush`,
`finalize`, `reset`, `getState` and optional `dispose`. Preserve
`revision`, `committedText`, `previewText` and finalization semantics across
chunk boundaries. Test reset during in-flight work, stale callbacks, empty
flush/finalize, cancellation and repeated reuse; do not claim native
incremental decoding merely because this shared streaming facade is present.

Develop alignment, word timestamps, diarization and other useful speech
capabilities where justified. Separate ASR text, forced alignment and speaker
assignment contracts. Evaluate overlap handling and speaker-turn behavior,
not just readable text. Measure first-partial/end-of-utterance latency,
boundary loss, duplication, revisions, drift and long-session memory.

## Examples are part of the library's acceptance system

Inspect the parent workspace `N:\github\asrjs`, not only this repository.
First-class consumers and reference projects include `benchmark-demo`, `browser-demo`, `playground`,
`streaming-demo`, `vad-demo`, `firered-vad-web`, `webgpu-agent-test`, and
`parakeet-ui-webgpu`. They exercise model management, API/model usage,
streaming, VAD, monitoring, benchmarking and hybrid WebGPU/WASM execution.

Inspect each app's actual dependency/alias wiring and public API usage rather
than assuming every sibling already uses the package in the same way.
`firered-vad-web` is a separate VAD reference; its existence does not establish
an integrated FireRed ASR model. Legacy Parakeet UI behavior is a compatibility
reference, not authority to reintroduce its old architecture into the core.

Keep examples small, correct, current and executable through public library
APIs. Move proven reusable headless behavior into core, leaving application UI
and framework glue outside. Browser harness pages must call library inference;
they must not contain a second decoder used to certify the first.

For each change identify affected companions and run relevant builds/smokes/
browser loops. Preserve independent Git states and respect writable boundaries;
request authority when a needed sibling edit is outside the permitted scope.
Report untested companion integration explicitly, not as implied completion.

## Model acquisition, reference and porting workflow

For each serious candidate, preserve this evidence chain:

`original weights + official engine -> numerical capture -> own fp32 ONNX ->
native ORT comparison -> library WASM/WebGPU/native paths -> real applications`

1. **Audit the original model.** Verify official/upstream-recommended weights
   and inference engine, architecture, task, languages, license, tokenizer,
   processors, decoding settings and resource requirements. Pin revision and
   SHA-256. Obtain the original PyTorch/NeMo/etc. weights, not merely a community
   ONNX/GGUF file. Reuse verified local assets; respect download/storage policy.
   Keep large weights and generated traces out of Git. Record provenance.
2. **Run the original engine.** Generate an implementation oracle from those
   original weights on deterministic fixtures: speech, silence, speech/silence,
   long audio and relevant languages. Capture inputs/features, encoder outputs,
   caches, predictor/decoder states, logits, decoding decisions, tokens,
   timestamps and text where applicable. Instrument the real inference path
   and check that instrumentation does not alter its output. A generic wrapper
   or another ONNX port is not automatically an equivalent oracle.
   Bind the oracle to effective model-input PCM, not only the source file:
   record decoded sample rate/count/channels, waveform dtype/hash, resampler and
   its settings, scaling/channel mixing, padding and window/chunk policy.
   Require equivalent effective inputs before treating later differences as
   model parity failures. Shared canonical PCM can isolate model math, but
   does not replace separate acceptance of real decoding/resampling paths.
3. **Export our own fp32 graphs.** Study our verified Parakeet and Whisper
   exports, especially `tools/whisper-onnx-export/` and its four-graph pattern.
   Choose family-appropriate boundaries, not four graphs for every model.
   Preserve actual layouts, dtypes, dynamic lengths, state/cache semantics,
   tokenizer IDs, blank/EOS rules and colocated external data. Validate the
   artifact contract; keep exports/transformations reproducible and reversible.
   Audit external tensors throughout the graph, including tensor attributes and
   nested graphs, not just initializers. Validate sidecar hashes, ranges and
   large offsets; byte-identical repacking is packaging evidence, not inference
   parity. Keep the unmodified source artifacts and transformation receipts.
4. **Establish numerical parity first.** Compare the same audio and processing
   contract stage by stage using explicit atol/rtol, max/mean absolute error,
   cosine/allclose, mismatch counts and first divergence. Fail on shape/count,
   missing/nonfinite data, source identity or one-sided metadata mismatches.
   Compare equivalent raw/normalized score boundaries explicitly; do not hide
   a normalization change by relaxing tolerances. Require exact token/text
   behavior and applicable duration/frame/cache checks. Debug the earliest
   divergence before downstream symptoms. Shape or plausible text is not parity.
5. **Integrate through the library.** Use an existing family where valid and
   a thin new preset/source configuration as needed. Run the public API with
   real artifacts, repeated calls, errors, cancellation and disposal. Validate
   native ORT, ORT Web WASM, real browser WebGPU and hybrid placements separately
   where supported. Record unsupported combinations; do not fake universal
   support or silently fall back while reporting WebGPU/native execution.
6. **Optimize and revalidate.** After fp32 token/text parity, evaluate fp16
   and rerun the same oracle. Optimize graph partitioning, memory/VRAM, CPU/GPU
   transfers, cache/state placement, loading and warm latency with matched A/B
   controls. Integer quantization is not a default: consider it only when size
   otherwise prevents a useful port, as a separate evidence-gated decision.
   Previous int4/int8 results are not a reason to quantize every new model.
7. **Assess quality and promote carefully.** Measure WER/CER on appropriately
   held-out audio using consistent normalization; original-engine output is
   an implementation oracle, not human ground truth. Keep training/evaluation
   overlap and model-card claims explicit. Validate affected examples, document
   licenses, languages, limits, backends and artifact recipes before promotion.

Keep canonical audio, human labels and official oracle captures immutable.
Write candidate outputs, diagnostics and transformations to separate locations.
An inapplicable parity rung needs a family-contract explanation; a difficult or
missing rung is FAIL/UNTESTED/BLOCKED, not silently removed from acceptance.

Seek the best measured CPU/GPU/VRAM trade-off for this library, not an
unsubstantiated claim of globally optimal graphs. Retain fp32 controls and
reject a speedup that changes required output semantics or leaks resources.
Third-party sherpa/C++/GGUF recipes are useful engineering evidence and
secondary controls, not substitutes for our original-weight reference chain.

For an optimization slice, name the target host/device, workload, graph and
precision before measuring. Compare supported graph partitions and component
placements, including WebGPU encoder with WASM decoder/joint where applicable.
Record matched before/after correctness, cold load, warm latency/RTFx, peak
CPU/RSS/WASM memory, available VRAM measurements and transfer/allocation costs.
Keep a faster variant only when required parity and lifecycle checks still
pass, and document trade-offs rather than hiding one memory cost behind another.
Set practical acceptance budgets from actual baselines; do not invent universal
performance promises or call a graph optimized just because ORT can load it.

## Improve the engineering system while doing the work

Follow `MODEL_HARNESS_CONTRACT.md`. Reuse the existing smoke tests, simulations,
reference-capture tools, stage comparator and real browser WebGPU loops. Every
discovered hardcode, false PASS, unsafe fallback or family leak should lead to
a small fix plus a focused regression when it is in the slice's scope.

Existing starting points include:

- `tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py`
  for original NeMo inference/capture, and
  `tools/nemo-tdt-onnx-export/` for family-owned export and recorded-stage replay.
- `tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs` and
  `tests/smoke/transcribe-fixture.mjs` for public-library artifact runs.
- `examples/nemo-tdt-native-local.mjs` for a runnable public-API Node/Bun
  consumer, with `tests/smoke/nemo-tdt-native-example-smoke.mjs` exercising that
  actual CLI through the existing oracle adapter rather than a second decoder.
- `tools/model-debugging/scripts/node-compare-stage-captures.mjs`,
  `tools/tensor_parity.py` and `tests/support/fixture-gates.ts` for shared
  comparison/prerequisite machinery.
- `tests/nemo-tdt-*.test.ts`, `tests/pipeline-windowing-stage.test.ts` and
  `tools/model-debugging/reference/hf-parakeet-port/window-overlap-sim-test.mjs`
  for focused family and windowing simulations.
- The sibling `../webgpu-agent-test` for real browser WebGPU/hybrid loops,
  and the existing Whisper export/test tools for proven graph and cache work.

These are starting points, not a promise that every runner already supports
every candidate or host. Inspect their actual CLI/API and extend existing
adapters where needed; do not invent commands or duplicate inference logic.

Share model-neutral identity/hash checks, fixture gates, numerical metrics,
timeouts, receipts, orchestration and cleanup. Keep family adapters and tests
responsible for required stages, graph names, layouts, decoder rules,
tokenizers, score normalization and tolerances. Model-specific assertions
remain model-specific. No model-ID branches in a supposedly generic runner,
no universal decoder invented before real reuse, and no duplicate oracle loop.

Use data-driven candidate/artifact/audio/oracle selection and configurable
paths. Optional tests must genuinely SKIP when not requested; a requested
missing/empty/malformed artifact must FAIL. Simulated sessions prove scheduling,
state/shape/lifecycle logic, not numerical parity or GPU execution. Add real
artifact checks separately, with backend identity and machine-readable output.

Start with syntax/types and targeted tests, then family/component tests,
cross-family regressions for shared changes, package/root import checks,
native/WASM/browser runs and performance/soak tests. Fix diagnostic scripts
before trusting their findings. Keep testing/tool dependencies outside the
published runtime and distinguish mocked from real acceptance evidence.

Maintain separate evidence for these lanes; one lane cannot certify another:

| Lane                                    | What its successful run establishes                                                          | What it does not establish                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Unit tests and simulated sessions       | Contracts, scheduling, cache/state transitions and exercised cleanup paths.                  | Real numerical parity, real GPU use or ASR quality.                      |
| Original engine capture                 | Pinned official inference behavior on identified audio, with instrumentation controls.       | Human transcript truth or correctness of our port.                       |
| Native Python ORT recorded-stage replay | Exported graph behavior on the captured inputs and states.                                   | Independently accumulated library decoding or Node/Bun support.          |
| Public library in Node native ORT       | Actual Node engine/provider, features, accumulated output and exercised lifecycle.           | Bun acceptance, browser execution or untested native providers/families. |
| Public library in Bun native ORT        | The independently exercised Bun configuration and output.                                    | All Bun versions, all models or all native providers.                    |
| Public library in ORT Web WASM          | The requested WASM graph, processor and output contract.                                     | WebGPU execution or native ORT execution.                                |
| Real browser WebGPU and hybrid loops    | Actual adapter/component placement, numerical/output checks and exercised browser lifecycle. | Other devices, precisions, graph variants or unmeasured VRAM.            |

Each real run should identify the source revision and weight/artifact/config/
audio/oracle hashes, actual host and engine versions, component providers,
precision, decoding options, commands, timings and status for every required
check. Retain full sequences for comparison; checking only the first 30 tokens
or readable text is insufficient. Emit a success receipt only after required
comparisons and awaited cleanup succeed. Never let a stale result file or a
receipt written before disposal hide a failed run.
Identify the executable build actually used, not just the source checkout:
record compiled artifact identity and every relevant graph/sidecar. A source
revision alone does not prove which build ran, and compiled hashes alone do
not prove source/build equivalence. Bind the reported acceptance to both when
possible and refresh affected runs after changes.

For hybrid execution, record requested and actually exercised engine/provider
and precision per component/graph, not just one model-level backend label.
Classify differing or unproven effective audio preparation as an input-contract
failure before issuing a model-parity verdict. Keep that failure visible and
add a same-input control; never erase it by selecting an easier fixture.

Improve graph inspection and compatibility reports when recurring failures
justify it: actual I/O contracts, dtypes, shapes, opsets, operator use, external
data, estimated memory and version-specific ORT/WebGPU risks. Static analysis
predicts risks; only real execution establishes support. Record a reusable
failure category, earliest failing stage, environment, reproduction command and
next discriminating experiment rather than merely saying "model failed".

Evaluate custom WGSL kernels, graph transforms or family scaffolding only when
profiling or repeated porting work justifies them. Preserve parity controls and
keep architecture-specific kernels and adapters in their proper boundaries.
Update relevant repository playbooks/skills with the symptom, cause, tested
recipe, limits and regression evidence. Do not invent tool commands or folders
that have not actually been implemented.

Track whether the system is improving: time to a valid export and first
divergence, manual graph edits, reusable checks, fixture coverage and repeated
agent effort. A substantial investigation should leave a useful regression,
recipe, diagnostic improvement or failure record; it need not manufacture an
abstraction when no reusable pattern has been proven.

## Delivery tracks and persistent tasks

Maintain a whole-product backlog as well as the model queue. Use the tracks
below to organize bounded implementation slices; they are acceptance targets,
not claims that these capabilities are already complete. Extend the existing
task records rather than keeping contradictory plans in multiple documents.

| Track                                      | Required outcome                                                                                                                   | Evidence before calling a slice done                                                                                                                                     |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| G0 — Establish the baseline                | Reconcile architecture, worktrees, current ports, examples and test coverage.                                                      | An evidence-backed status map, preserved dirty work and a prioritized next task.                                                                                         |
| G1 — Deliver verified models               | Complete Orukeet, then audited small Parakeet distills, then ranked non-NVIDIA candidates.                                         | Original-weight provenance, official oracle, fp32 numerical/token/text parity, real public API and supported browser/native paths, quality checks and documented limits. |
| G2 — Make the API easy                     | Improve loading, capability discovery, transcription, realtime results, progress, cancellation and disposal.                       | Typed public-API examples, actionable error tests, stable canonical outputs, migration notes and affected consumer checks.                                               |
| G3 — Make the runtime portable and compact | Improve assets, external data, caching, workers, lifecycle and separately verified Node/Bun native execution.                      | Browser-safe imports/package smoke, actual engine/provider receipts, partial-load/cancel/reset/dispose checks and measured size/memory/startup behavior.                 |
| G4 — Make speech workflows dependable      | Improve VAD, streaming/windowing, long audio, timestamps, alignment and diarization where useful and supported.                    | Speech/silence/boundary fixtures, long-session checks, partial/final semantics and task-specific quality/latency measurements; unsupported capabilities remain explicit. |
| G5 — Keep examples usable                  | Maintain companion apps as executable demonstrations of supported public workflows.                                                | Relevant build/smoke/browser results, correct model management/hybrid usage and no duplicate application decoder or core lifecycle logic.                                |
| G6 — Improve the engineering system        | Remove recurring hardcodes and strengthen export, comparison, simulation and browser automation while delivering the other tracks. | Focused regressions, reproducible receipts and a tested recipe or failure record that reduces repeated effort.                                                           |

Start G0 by refreshing live state, not by rerunning every expensive test. G1
keeps the user-confirmed candidate order. G2-G6 support it and also admit
independent, high-value product increments: do not postpone all API/runtime/
example work until every model is ported, or reorder models because an unrelated
refactor is more convenient. Dependency fixes can precede the model step that
requires them, with the reason and affected acceptance checks recorded.

For each persistent task record its ID, user-visible outcome, affected files
and companion apps, dependencies, baseline, acceptance matrix, status,
reproduction commands, evidence paths and next smallest experiment. Split large
tasks into reviewable slices. Keep failed and skipped checks visible. A shared
fix needs cross-family tests; a model-specific fix belongs with its adapter and
must not silently change another family's processor or decoder behavior.

Keep this proportionate to risk. A documentation or small API/example slice
needs its outcome, focused diff/checks, affected consumers and honest remaining
limitations; it does not need a model-weight receipt or every backend lane.
Real model/artifact/backend slices require the full relevant identity,
numerical/output, lifecycle and environment evidence. Do not turn the reporting
system into a prerequisite for every useful product improvement.

Use this compact task structure within the existing backlog, expanding the
acceptance section only for the relevant slice:

```text
ID / track / status:
User-visible outcome:
Dependencies and baseline:
Owned files / affected public workflows and companion apps:
Acceptance checks: [each required check and PASS/FAIL/PARTIAL/UNTESTED/SKIP]
Commands and evidence:
Measured change and remaining limitations:
Next smallest actionable step:
```

Give each substantial investigation a proportionate time/compute/storage budget
and explicit exit criteria; keep routine product edits lightweight. Prefer
discriminating experiments over repeating an unchanged
full pipeline or collecting more traces without a decision. If the budget is
exhausted, preserve the unresolved gate and evidence, continue independent
agreed product work, and request direction for any material candidate-order or
scope change. Do not silently turn a blocked numerical rung into acceptance.

Close a delivery milestone with the implemented outcome, executable public-API
example or recipe, relevant regression/consumer results and documented limits.
Choose those checks by the changed workflow, not by requiring every model lane
for a small product fix. Keep the next resume point in the existing task record;
these milestones do not introduce another work queue.

### Initial delivery milestone

Use the following as the recommended first delivery scope, mapping each outcome
to the existing G0-G6 tracks and P01-P10 tasks rather than creating another queue.
Resolve the exact fixtures, supported host/provider/precision matrix and
performance budgets from the live baseline before implementation. These are
future acceptance targets, not statements that current code already passes.

- **Verified Orukeet path:** finish the original-weight/original-engine to own
  fp32 graph to public-library chain, including numerical and full-output
  checks, representative audio, required native Node/Bun and real browser
  WebGPU/hybrid runs, quality evidence and resource cleanup. Keep unpassed
  precision or provider variants explicit; do not promote fp16 before fp32.
- **Usable API and examples:** show at least two established model families
  using the same canonical runtime/session contract; deliver small executable
  browser file/realtime and native local-artifact examples. Validate affected
  companion model-management, streaming and hybrid workflows through the public
  library. Examples must not contain another model executor or decoder.
- **Compact and dependable runtime:** establish package/install/bundle/startup
  and inference-memory baselines, deliver a measured justified improvement,
  and retain browser-safe imports plus loading, cancellation, switching and
  awaited-disposal checks. A dependency packaging change needs clean-consumer
  validation; lazy native imports alone do not prove a small installation.
- **Reusable engineering and next candidates:** retain passing neutral harness
  regressions and separate family-specific tests, reproducible export/oracle/
  browser recipes and useful failure records. Audit genuine smaller Parakeet
  students and rank the next non-NVIDIA speech leads with source evidence.
  Do not require every future research candidate to be ported before this
  delivery closes, or describe an unverified candidate as supported.

If a numerical investigation reaches its stated budget, preserve its failing
gate and next discriminating hypothesis, then continue independent agreed API,
runtime or example work. A diagnostic should enable a concrete implementation
or feasibility decision; more traces alone are not a delivery outcome. Any
change to a required result or the confirmed model order needs user direction.

## Current priorities and honest continuation checkpoint

Follow `PARAKEET_VARIANT_TASKS.md` for P01-P10. The user-confirmed order is:

1. Orukeet on the existing NeMo TDT / Parakeet v3 path.
2. Genuine smaller Parakeet student/distilled checkpoints, then useful derivatives/fine-tunes
   after auditing architecture, tokenizer, quality and license. Smaller files
   caused by fp16/INT8 packaging are not smaller student models.
3. High-value new non-NVIDIA ASR and related speech candidates. Revisit the
   dated candidate report and original FireRed/Qwen3-ASR 0.6B interests; research
   Moonshine, Fun-ASR, Cohere, Voxtral, ARK-ASR, alignment and diarization as
   candidate hypotheses, not pre-approved or completed ports. Verify current
   official sources and record an evidence-based ranking before implementation.

The user's dated research list is a discovery seed, not a verified catalog.
Also revisit Phonon, new GigaAM variants and on-device alignment candidates
such as `desert-ant-labs/align` where useful. Resolve exact upstream identities
and revisions and check whether this repository already has relevant partial
work before proposing another implementation. An existing Node-, C++-, TFLite-
or CoreML-only implementation can inform our browser port; its lack of a ready
browser package is not itself a rejection reason. Assess the real weights,
operators, conversion route, license, resource limits and expected user value.

Rank candidates by expected quality, browser feasibility and strategic value,
including Turkish/multilingual usefulness, held-out accuracy, latency, peak
memory, actual parameter count, tokenizer/decoder cost, licenses and effort.
Research notes and leaderboard claims are hypotheses to verify, not local
acceptance results. Separate a true smaller student from a fine-tune or a
smaller precision/packaging variant. Start with a bounded feasibility experiment
and maintain reasons for advancing, deferring or rejecting each candidate.

Do not start new NVIDIA model work beyond existing Parakeet TDT v2/v3 for now.
Nemotron 3.5 Streaming ASR is paused at partial browser parity; preserve its
exports and negative results. It must not block the queue or be called a
completed/bad model. A browser port can be deferred after a bounded,
evidence-backed feasibility failure without lowering acceptance criteria.

Latest scoped resume update as of 2026-10-01:

- Signed filterbank-weighted power attribution has now run in
  `tools/data/results/orukeet/library-feature-weighted-power-jfk-2026-10-01.json`.
  All four aligned candidate raw-mel reconstructions are exact. The controlled
  Float32/Float64 FFT paths still have 129/120 normalized-feature mismatches;
  the official-complex controls have zero. The production path still has 313
  mismatches and was not changed. This is **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**.
  The next discriminating experiment is an independent direct-DFT comparison
  at actual failing frame/bin supports, not another statistics substitution
  or rerun of the completed weighted-power attribution.
- NeMo TDT initialization now retains original acquisition/disposal errors
  together with rejected cleanup errors; six focused lifecycle cases include
  synchronous/asynchronous release failure and disposal during encoder load.
  Real Node/Bun Identity-graph controls report **PASS_RESOURCE_ONLY** in
  `tools/data/results/harness/nemo-tdt-{node,bun}-native-resource-final-2026-10-01.json`.
  This proves the scoped acquisition/release path, not ASR numerics, browser
  cleanup, cancellation/soak behavior or all-family resource ownership.
- The refreshed public CLI JFK controls,
  `public-example-{node,bun}-jfk-cleanup-race-verified-2026-10-01.json`, match
  the full 38-token output with awaited cleanup and remain
  **PARTIAL_PASS_OUTPUT_ONLY**; legacy effective PCM is still **UNVERIFIED**.
  The full-suite report
  `tools/data/results/harness/weighted-power-cleanup-verified-full-suite-2026-10-01.json`
  records 1,290 passes / 27 skips / 0 failures, not numerical model acceptance.
  Earlier speech/silence and package receipts describe their recorded source
  revisions; reconcile freshness before reusing them after later code edits.

The following snapshot preserves the earlier evidence ladder. The scoped
update above supersedes its historical next-step wording, not its failed gates.

Recorded local Orukeet snapshot as of 2026-10-01:

Receipt basenames in this checkpoint refer to `tools/data/results/orukeet/`.
They describe the recorded run, not an automatic certification of later code.

- Original `.nemo` acquired and hashed; source config and all 8,193 indexed
  tokenizer IDs match v3. Orukeet retains about 627M parameters; it is not a
  small distilled model.
- Official NeMo JFK and speech/silence captures exist with exact before/after
  instrumentation controls. Pure silence and long-audio captures remain open.
- Two own fp32 graphs were exported using the family exporter under
  `tools/nemo-tdt-onnx-export/`. The original export receipt says
  **EXPORTED_UNVALIDATED**; subsequent recorded-boundary replay evidence below
  is a separate partial validation, not port completion. An initializer-only
  audit missed a `Constant` attribute: the
  encoder actually has 295 external tensors. They are now byte-identically
  consolidated in a fresh directory for the library's single-sidecar loader.
- Native Python ORT CPU replay passed full encoder/length, fused-head states,
  normalized scores and per-recorded-call token/duration argmax for JFK and
  speech/silence (916 numerical comparisons, 228 joint calls). This reuses
  captured inputs, not an independent decode loop. Predictor raw output is not
  exposed; this does not certify the downstream public API or browser.
  See `native-fp32-stage-replay-2026-10-01.json`.
- Stock v3 real-browser hybrid control passed; it is not Orukeet validation.
- The default built-in composition still does not enable native inference.
  Current uncommitted source adds an explicit experimental `native-cpu`
  backend with family opt-in and NeMo TDT plumbing; it is not general native
  support or a released compatibility promise.
- Separate real Node and Bun public-library fp32 JFK runs produced the same
  full 38-token sequence, text, frame indices and TDT durations as official
  NeMo, using packed artifacts and JS features. Refreshed oracle-bound receipts
  are `library-node-native-fp32-jfk-oracle-2026-10-01.json` and
  `library-bun-native-fp32-jfk-oracle-2026-10-01.json`; both record
  **PARTIAL_PASS_OUTPUT_ONLY**, actual host/ORT versions, artifact hashes and
  successful awaited cleanup. These recorded Node 26.2.0 and Bun 1.3.14 runs
  are one-fixture exact-output results, not complete numerical parity, broad
  native compatibility, quality evaluation or browser support. Refresh any
  affected acceptance after subsequent source changes.
- The original 22.05 kHz speech/silence public-library run recorded 69 official
  versus 76 candidate tokens, with the first token mismatch at index 34:
  `library-node-native-fp32-speech-silence-oracle-2026-10-01.json` remains
  **FAIL_OUTPUT_COMPARISON**. Its inspector used linear resampling, while the
  official NeMo path uses Lhotse's windowed-sinc resampling. The raw WAV hash
  does not prove equal effective PCM; this run is not a valid same-input model
  parity verdict and does not establish a decoder fault. Preserve this failed
  receipt and the original-rate fixture for separate resampling acceptance.
- The official engine has since captured the actual preprocessor-input PCM
  in `official-speech-silence-waveform-stages-2026-10-01.json.gz`, with
  `official-speech-silence-waveform-summary-2026-10-01.json` recording all six
  unchanged-output instrumentation controls. It contains 649,600 mono float32
  samples at 16 kHz, no right padding, and the f32le SHA-256
  `8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
  Separate real public-library Node and Bun runs using this exact captured PCM
  match all 69 tokens, frame indices, TDT durations and the full official text;
  awaited cleanup also passes. See
  `library-node-native-fp32-speech-silence-same-pcm-2026-10-01.json` and
  `library-bun-native-fp32-speech-silence-same-pcm-2026-10-01.json`, both
  **PARTIAL_PASS_OUTPUT_ONLY**. This supports an input-preparation explanation
  for this fixture's old 69-versus-76 difference; it is not numerical feature
  acceptance, an accepted production resampler, broad native compatibility or
  Orukeet browser validation.
- The existing inspector now offers explicit `--input-mode oracle-pcm` for
  this frozen-input diagnostic, with source/oracle bindings, float32/shape/
  length/hash checks and a recheck of the public PCM container. Source-WAV
  oracle comparisons reject incompatible rate/channels or different captured
  PCM before model inference. Legacy captures without effective-input data
  remain **UNVERIFIED** on that input rung; output agreement cannot turn them
  into waveform parity. Padded captured inputs are currently rejected by this
  diagnostic rather than trimmed silently. Its focused tests do not replace
  independent acceptance of normal file decoding/resampling paths.
- A public-API local native consumer example is now delivered separately from
  the inspector: `examples/nemo-tdt-native-local.mjs`. Its real CLI was exercised
  independently in Node 26.2.0 and Bun 1.3.14 with native ORT CPU, complete
  canonical/native output and awaited runtime-owned cleanup. The four
  `public-example-{node,bun}-{jfk,speech-silence}-verified-2026-10-01.json`
  receipts match the complete 38-token JFK and 69-token speech/silence outputs,
  frame indices, TDT durations and text. They remain
  **PARTIAL_PASS_OUTPUT_ONLY**: JFK effective PCM is UNVERIFIED; speech/silence
  uses the exact captured official PCM in a derived WAV, not accepted normal
  source decoding/resampling. Pure WAV decoding is shared between browser and
  native consumers without changing production feature math. The receipt
  wrapper reuses existing identity/input/output checks, hashes both graph
  sidecars and the complete actual compiled build, and writes success only
  after comparison and cleanup. The source-bound delivery handoff is
  `tools/data/results/harness/native-local-example-handoff-2026-10-01.json`,
  status **DELIVERED_NATIVE_CONSUMER_SLICE_MODEL_PARTIAL**. Its recorded checks
  are 75 focused passes and 1,283 full-suite passes / 27 skips / 0 failures;
  these do not close the model's numerical, browser or quality gates. Its own
  package baseline is 1,092,517 tarball bytes / 5,149,036 unpacked bytes /
  1,401 files, excluding weights and installed engine dependencies. This is a
  baseline, not an optimization claim. Reconcile the handoff bindings before
  reuse; extend missing workflows instead of reimplementing this example.
- Full JS feature parity on JFK still FAILS at the existing
  `atol=1e-5, rtol=1e-4`: 313 of 140,928 values mismatch, maximum absolute
  error `3.4236907958984375e-4`. The official processor reproduction is exact;
  applying JS normalization to the same official raw mel passes. An isolated
  float32-preemphasis experiment reduces mismatches to 200 but still FAILS.
  It is diagnostic only, not an accepted production fix. See
  `library-feature-boundary-jfk-2026-10-01.json` and
  `library-feature-preemphasis-experiment-jfk-2026-10-01.json`.
- Further isolated, weight-free experiments align float32 preemphasis,
  the official window and float32 window multiplication and reduce the
  mismatch count to 121, still FAIL. Injecting the official power spectrum
  into the JS mel/log/normalization path then passes with zero mismatches
  and maximum absolute error `3.5762786865234375e-6`. This localizes the
  remaining controlled discrepancy upstream of mel/log/normalization; FFT/power
  arithmetic is the next boundary to investigate. Production JS arithmetic has
  not been changed or accepted by this experiment. See
  `library-feature-power-experiment-jfk-2026-10-01.json`, whose status is
  **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**.
- Direct power-spectrum comparisons in isolated aligned controls pass their
  allclose threshold, but their downstream normalized features still have 121
  mismatches. Rounding final complex outputs to float32 reduces that to 120,
  still FAIL with maximum absolute feature error `2.5162100791931152e-4`.
  Direct power tolerance therefore does not certify downstream features;
  investigate FFT arithmetic/order with a discriminating control, not another
  unproven production patch. See
  `library-feature-power-boundary-jfk-2026-10-01.json` and
  `library-feature-complex-power-experiment-jfk-2026-10-01.json`. These remain
  diagnostic experiments; production JS arithmetic has not been changed.
- A matched full 512-point FFT experiment now uses Float32 versus Float64
  scratch on identical float32 windowed input and the existing compiled FFT
  helper. Independent direct-DFT and capture/immutability/guard tests pass.
  The real JFK Float32 control leaves 129 feature mismatches, max error
  `3.0434131622314453e-4`; Float64 leaves 120, max
  `2.5162100791931152e-4`, like the earlier final-complex-rounding control.
  Both power-spectrum allclose checks pass, but normalized features still FAIL.
  This is not a production fix. See
  `library-feature-full-fft-control-jfk-2026-10-01.json`; it motivated the exact
  framed-input complex comparison recorded below, not a production replacement.
- That comparison now captures canonical f32le frame bytes. A bit-identity
  regression caught JSON numeric capture losing `-0`. After fixing the diagnostic,
  both full FFT controls match all 563,712 official windowed input values bit-for-
  bit, including matching SHA-256; official framed Torch RFFT reproduces NeMo's
  complex STFT exactly. JS complex components pass existing allclose on those
  same bytes, but feature parity still FAILS at 129/120 mismatches. See
  `library-feature-complex-fft-f32le-jfk-2026-10-01.json`. Fourteen probe tests
  pass and production processor hashes are unchanged. The earlier JSON-only
  complex receipt remains diagnostic failure evidence, not input-bit acceptance.
- The next official-complex injection control has already run. Supplying the
  actual official complex STFT components through canonical float32 bytes to
  either JS power formula produces features with zero mismatches and max error
  `3.5762786865234375e-6` at the existing tolerances. Power passes those
  tolerances but is not exactly identical: the exact diagnostic records
  153,688 differing values for the ordinary formula and 1,862 for the
  float32-rounded control. Ordinary production features still FAIL at 313
  mismatches, and the full-FFT controls still FAIL at 129/120. See
  `library-feature-official-complex-injection-jfk-2026-10-01.json`, status
  **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**. This narrows the controlled discrepancy
  to FFT-derived power for this fixture; it does not prove an accepted FFT fix,
  production feature parity or whole-port correctness. Do not repeat this
  completed injection experiment as the next unexplored step.
- A raw/statistics attribution diagnostic now captures candidate raw mel before
  normalization/masking, uses actual official float32 mean/std and retains all
  mismatch coordinates plus explicit arithmetic residuals. See
  `library-feature-normalization-decomposition-jfk-2026-10-01.json`;
  28 focused probe tests pass. Production remains 313 mismatches versus 314
  using official statistics counterfactually; full-FFT controls remain 129/120
  versus counterfactual 131/121. Both official-complex controls remain zero.
  Reconstruction checks pass; four-term closure is below `8.51e-16`.
  Statistics substitution does not fix this fixture. The statistics term also
  includes engine arithmetic; do not claim a pure statistical causal split.
  These are diagnostic results, not model acceptance; tolerances and production
  source/compiled processor hashes are unchanged. Failing coordinates include
  both low and high mel bands. Next correlate signed filterbank-weighted power
  differences and active FFT-bin support, not a low-frequency-only hypothesis.
- The uncommitted public inspector now includes streamed artifact hashes,
  original-weight/audio oracle bindings, actual engine/host versions, full TDT
  output comparisons and awaited cleanup before writing a success receipt.
  The refreshed JFK and same-PCM receipts above exercise recorded versions of
  that implementation. Its receipt and PCM-contract tests cover full mismatch
  reporting beyond token 30, same-rate mono prerequisites and corrupt or
  mismatched effective-input metadata. Validate
  changed scripts and strict test typing before trusting later receipts;
  core typechecking alone does not cover every test file, and historical suite
  results do not certify later edits or the whole model.
- The shared fixture-gate migration now also covers the Whisper reproducibility
  harness. Its metadata/feed typing passes separate strict compilation, and
  five actual-runner regressions distinguish disabled inference SKIP from
  requested missing/empty/malformed reference and missing mel/audio FAIL.
  Its existing token thresholds are unchanged. The full regression receipt
  `tools/data/results/harness/fixture-gates-full-suite-2026-10-01-repro.json`
  records 1,192 passed, 27 skipped and no failures. This includes correcting
  two earlier false PASS cases to real SKIP; it is not model parity evidence.
- Session setup now also uses test-only model-neutral ownership/rollback.
  Nine helper checks and two actual Whisper harness setup simulations cover
  acquired-resource cleanup and retry after rejected cached setup; the latter
  two failed before wiring. Strict compilation passes. The fresh full-suite
  receipt `tools/data/results/harness/fixture-session-cleanup-full-suite-2026-10-01.json`
  records 1,203 passed / 27 skipped / 0 failed. Existing numerical/token criteria
  remain unchanged; real Whisper model inference is not established by this run.
- A separate production helper, `src/runtime/owned-session.ts`, now tracks a
  session before initialization and rolls back unreturned failed creation.
  NeMo TDT and Whisper use it without moving their engine/decode logic into
  runtime. Fourteen simulated helper/family tests now also cover post-disposal
  creation, initialization racing disposal and repeated disposal awaiting the
  existing cleanup. Four new cases failed before the guards/memoized cleanup
  fix. The fresh full-suite receipt
  `tools/data/results/harness/owned-session-race-full-suite-2026-10-01.json`
  reports 1,217 passed / 27 skipped / 0 failed; focused checks passed 45 tests.
  Browser import and package dry-run guards pass. Rebuilt Node/Bun JFK runs,
  `library-node-native-fp32-jfk-owned-session-race-2026-10-01.json` and
  `library-bun-native-fp32-jfk-owned-session-race-2026-10-01.json`, match all 38
  tokens, frames, durations and text with cleanup PASS. They remain
  **PARTIAL_PASS_OUTPUT_ONLY**, not feature or legacy effective-input parity.
  All-family lifecycle, actual failed partial ORT acquisition and browser
  release remain separate gates; refresh affected wider-fixture acceptance
  after lifecycle edits. Historical suite totals do not certify later code.
- Partial-load, async abort/disposal and preprocessor-output cleanup regression
  tests now exist. Simulated lifecycle checks do not replace repeated real
  model runs or cross-family regression coverage. Pure silence, long audio,
  multilingual quality, real Orukeet WASM/WebGPU/hybrid and fp16 acceptance
  remain open. The JFK diagnostic config's English label is not authoritative
  full-model language metadata or a ready public preset.

Refresh this checkpoint from live files before relying on it. Next model step:
investigate the earliest feature divergence with the existing weight-free
`compare_js_feature_boundaries.py` diagnostic, separating preemphasis, window,
FFT, mel projection and normalization rounding. On the now-verified controlled
frames, use the completed raw/statistics and signed weighted-power attribution
to select actual failing mel-bin/frame coordinates and active FFT-bin support
in both low and high mel bands. Compare those canonical-frame complex values
with an independent direct DFT to distinguish mathematical FFT error from
official float32 FFT arithmetic/order. Keep the completed official-complex
injection as a control, not an unexplored next experiment. Do not repeat
stats-only substitution or attribute the error to low-frequency cancellation
without a discriminating test. Use a focused
regression, not a blanket hardcode, tolerance relaxation or a
downstream decoder workaround. Preserve the same thresholds and official input.
Any shared processor change needs other-family checks before it is accepted.

In the same bounded workstream, validate and finish the existing public
inspector's identity/version/cleanup and effective-input contracts; refresh real
Node and Bun when lifecycle or harness changes affect the recorded runs. Keep
recorded-boundary replay separate from independently accumulated decoding.
The scoped NeMo TDT/Whisper wrapper initialization/disposal race is now covered;
the real Node/Bun NeMo TDT resource-only control is also recorded above.
Verify remaining actual model cancellation/soak, browser and other-family
cleanup separately; do not repeat the recorded resource control without a
source or hypothesis change.
Do not confuse the test-only harness helper with production resource ownership.
The speech/silence same-PCM output control above is already recorded; verify its
provenance and affected-code freshness rather than blindly repeating it.
Complete independent source-audio/resampling acceptance, the official fixture
matrix and full public-library numerical/output evidence on silence,
speech/silence and long audio. Then run supported
WASM and real browser WebGPU/hybrid loops before fp16 or public promotion.
Continue the API/runtime/examples tracks alongside these model dependencies;
do not turn this feature investigation into a new model-only project goal.

### Initial resume checklist

Use this dependency-aware checklist to begin, then record the concrete tasks
and results in the existing persistent backlog. It is not a second set of
completion claims. Preserve P01-P10 for the model lane and G0-G6 for the whole
product; API/runtime/examples work can progress alongside model dependencies.

1. **G0/G6 — Reconcile and restore a trustworthy baseline.** Check the live
   checkout and outstanding edits, compare current diagnostics and receipt
   tests with the latest evidence, and run their smallest relevant checks.
   Do not blindly rerun model downloads/exports or rewrite inherited work.
2. **G1/G6 — Resolve the earliest Orukeet feature divergence.** Use the
   completed signed weighted-power attribution to choose actual failing
   frame/bin supports for an independent direct-DFT comparison on identical
   inputs. Add a focused regression for a newly established distinction;
   accept a processor change only after the original thresholds and
   affected-family controls pass. Keep downstream successes labeled partial.
3. **G1/G3 — Complete reference and native coverage.** Capture missing pure
   silence and long-audio controls, extend numerical evidence on the already
   identified speech/silence PCM and retain original-rate resampling tests.
   Refresh affected recorded runs after source changes. Run the actual library
   separately in Node and Bun; verify full output, numerical stages where
   applicable, repeated reuse, cancellation and cleanup with provenance receipts.
4. **G1/G3/G5 — Deliver the browser path through the existing family.** Finish
   thin variant/configuration and verified metadata, run WASM, then actual
   Orukeet WebGPU/hybrid loops via the existing browser harness. Validate the
   relevant model-management and streaming consumers, not a parallel decoder.
5. **G2/G3/G4/G5 — Deliver a bounded product improvement in parallel.** Choose
   the highest-value verified API/runtime/example gap from the live baseline:
   loading and switching, progress/cancel/dispose, batch usage, streaming/reset,
   local native usage or long-audio behavior. State its user journey and tests,
   preserve cross-family output semantics and update executable documentation.
   The local NeMo TDT native consumer slice above is already delivered with
   partial model evidence; verify and reuse it rather than starting it again.
6. **G1/G3/G5 — Optimize and assess promotion.** Only after the required fp32
   parity gates, measure graph/backend variants and fp16 against retained
   controls, evaluate held-out multilingual quality and lifecycle/memory, and
   document supported configurations and limits in the public API/examples.
7. **G1/G6 — Advance the confirmed queue.** Audit genuine smaller Parakeet
   students separately from fine-tunes and quantized packaging, then rank new
   non-NVIDIA ASR/alignment/diarization opportunities using verified sources.
   Reuse the improved tooling; carry forward a tested recipe or failure record.

## Working discipline, reporting and completion

Read current Git status/worktrees, relevant handoffs and actual APIs before
editing. Preserve user changes. Use the repository's surgical edit/diff/test
workflow and stop repeated speculative patching. Do not commit, publish,
upload models, message other user chats, install heavyweight toolchains or
change external settings without the necessary user authority.

Keep a persistent ordered task list: purpose, dependencies, acceptance checks,
result paths, PASS/FAIL/PARTIAL/UNTESTED and next experiment. Update the charter,
architecture/examples when the corresponding product decision changes; do not
turn them into contradictory copies of the whole log. Record small receipts
and reusable lessons, preserving useful failed experiments.

For each implementation slice:

1. Reconcile the active checkout, current instructions, backlog and evidence;
   read applicable skills and inspect actual APIs before making changes.
2. Select one bounded user-visible outcome, its dependencies, baseline and
   acceptance checks. Prefer the first actionable unfinished priority; explain
   any deviation without silently replacing the user-confirmed model queue.
3. Implement surgical changes and validate from small checks to real public API
   and affected-example runs. Use cross-family regressions for shared changes.
4. Record commands, artifact identities, environment/provider, numerical and
   performance results, cleanup behavior and explicitly skipped checks.
5. Update persistent tasks and give the next bounded step. Continue while work
   remains safely actionable; when authority or critical evidence is missing,
   report the precise blocker instead of guessing or weakening acceptance.

Report concrete work, actual tests, skipped rungs, numerical evidence, measured
before/after performance, remaining risks and the next bounded step. Separate
load time, warm RTFx, first-partial latency, RSS/heap/WASM memory, actual VRAM
and tensor transfers; an unavailable VRAM measurement is unknown, not zero.

“Best-in-class” is a continuous direction, not a claim that every candidate is
finished. A bounded task is done only when its stated acceptance checks pass.
Do not mark the overall goal complete while required model, runtime/API,
example or portability work remains. The desired outcome is both a better ASR
product today and a smaller, more trustworthy effort to add the next model.
