# Recommended goal — whole-product speech runtime

Updated: 2026-10-08. This is the durable, harness-independent goal, not a
claim of current model acceptance. Read [AGENT_ENTRYPOINT.md](AGENT_ENTRYPOINT.md)
for startup order and the current handover. Task records remain
[PRODUCT_TASKS.md](PRODUCT_TASKS.md) and
[PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md).

## How to use

Give the next agent this file and ask it to resume through the entrypoint.
The charter need not be copied into its prompt: repository-aware agents must
read [PROJECT_CHARTER.md](PROJECT_CHARTER.md), architecture and safety themselves.
Do not create a new scheduled or persistent goal merely because this document
exists. On activation, reconcile live evidence and execute a bounded slice.

The former combined goal/checkpoint is preserved verbatim in
[the 2026-10-01 archive](archive/GOAL_PROMPT_RECOMMENDED_2026-10-01.md).
Its historical next-step instructions are superseded by the current handover.

## Resume emphasis — 2026-10-08

### Long-running goal activation text — 2026-10-08

The user explicitly requested a long-running Codex goal. Initial replacement
attempts were rejected while the previous goal was paused and unfinished;
it was not falsely marked complete. Live `get_goal` now confirms an active goal
referencing `C:/Users/steam/.codex/attachments/4be656a5-5cd9-446b-94f2-d08a4bc58c79/goal-objective.md`.
That attachment contains the detailed whole-product goal. The summary below
is a portable activation/resume prompt, not another queue. Recheck live goal
state on continuation; do not create a duplicate goal.

```text
Build @asrjs/speech-recognition in N:\github\asrjs\speech-recognition into a compact, capable, accurate, fast, memory-efficient, easy-to-use multi-model speech runtime for browsers and native Node.js/Bun. Deliver verified library/API/runtime/example improvements, not only model research.

Read docs/AGENT_ENTRYPOINT.md and docs/GOAL_PROMPT_RECOMMENDED.md, safety, architecture, charter, workspace context and harness contract. Reconcile live Git/worktrees, inherited edits, original weights, builds and receipts. Use existing PRODUCT_TASKS.md G0–G6/C records and PARAKEET_VARIANT_TASKS.md P01–P10; no competing queue. Old prompts are background.

Preserve one speech-first, ESM-first, headless, framework-neutral TypeScript package with coherent public API and narrow exports. Shared orchestration/assets/audio/contracts stay generic; family math/decoding/state and tests stay model-specific; presets remain thin; UI/framework wrappers stay outside core. Improve model loading/management, cancellation/reset/disposal, streaming/long audio/VAD/timestamps, packaging and parent-workspace examples. WebGPU is browser-primary, with explicit per-component WASM/hybrid placement. Test native Node/Bun separately from browser WASM in Node. Reuse the WASM-thread helper: floor(hardwareConcurrency/2)+2, respecting lower requests and single-thread fallback.

Model lane: Orukeet first; then Parakeet TDT distills and quantized/faster derivatives, including Redux, Ultra and efederici int4; then recorded 2026 non-NVIDIA candidates: Moonshine streaming, Fun-ASR MLT Nano, ARK-ASR, Cohere, Voxtral; preserve FireRed/Qwen, Phonon/GigaAM/alignment/diarization leads. Parakeet CTC is on hold; Nemotron/unrelated new NVIDIA work remain paused. Full-size derivatives can offer measured benefit; small files are not small students. Integer/ternary research is not default adoption.

Ports require verified original weights/revision/config/tokenizer/license/hashes, official/recommended-engine oracle, effective PCM identity and stage captures. Model output is not human ground truth. Export own optimized fp32 graphs and prove numerical/token/text/state parity at unchanged explicit tolerances; then revalidate fp16 or separately justified quantization. Verify public API, lifecycle, actual native/WASM/WebGPU/hybrid, held-out quality and matched latency/memory. No load-only, simulation-only or one-transcript acceptance. Improve existing capture/export/smoke/simulation/browser tools and neutral identity/metrics/cleanup regressions; preserve family-specific rules and failed evidence.

Redux source/format preflight commit d82f587: 28 local tests/264 blocks pass, not inference acceptance. Next establish exact-weight original-runtime oracle and kernel/export contract, without repeating failed teacher substitution or transferring Photon speed claims to ORT. Keep Orukeet's open numerical gates visible while independent product work progresses. Qwen uses isolated U:/parallel-agents/qwen-fp16-wt; coordinate via its docs/reports/qwen-fp16-direct-result-2026-10-08.md. Read the latest message for resource windows: MAIN-001 ended; MAIN-002 covers bounded schedule/precision diagnosis. Browser/GPU needs separate coordination.

Define a finite first-delivery acceptance matrix in existing tasks: required Orukeet/public-API/native-browser/example workflows, measured compactness/lifecycle improvements, strengthened harness and audited next-model backlog. Future candidates need not all be ported. Record commands and PASS/FAIL/PARTIAL/UNTESTED; ask before changing required outcomes. Complete only when agreed delivery gates are satisfied.

Continue safe bounded implementation, validation and scoped local commits. Preserve unrelated changes and immutable sources/oracles/gold. No broad staging/reset/clean, push/upload, large downloads, installs/upgrades, sibling writes or global settings changes without authority. Delegate edits only into isolated writable checkouts; coordinate heavy jobs. Request missing authority instead of weakening gates.
```

### Latest model scope and evidence

Latest user clarification: Parakeet CTC is on hold. Resume the recorded 2026
candidate list and Parakeet **TDT** distills, quantized variants and faster
derivatives. Same-size derivatives remain eligible for investigation when they
claim useful speed, memory or quality gains. Follow the active P10 checklist in
[PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md), not CTC repair work or
historical blanket exclusions based on NVIDIA teacher lineage. This does not
authorize new unrelated NVIDIA models, integer defaults, weaker gates, large
downloads or automatic promotion.

Read [GOAL_MODEL_PORT_HARNESS.md](GOAL_MODEL_PORT_HARNESS.md) as the current
engineering focus, not a replacement mission or permission to change model order.
Start tool discovery at [tools/README.md](../tools/README.md); for model work read
the project-local [Model Debugging Skill](../tools/model-debugging/SKILL.md).
The 2026-10-03 handover is a historical baseline: reconcile later commits and
dated receipts instead of replaying its next-step instructions literally.

The earlier resume baseline was `9a9fa4a`; current owned tooling checkpoint is
`d82f587` (Redux SHA-bound packed preflight, 28 local tests, format-only acceptance).
Reconcile live HEAD before execution. Later evidence records a passing
Orukeet JFK hybrid score control after correcting the loaded ORT bundle identity;
production feature conformance remains FAIL / deferred at its unchanged gates.
Do not repeat that resolved score attribution or relax the feature gate. Existing
Moonshine WASM output controls and Qwen browser memory failures have narrower
artifact/host scopes; neither supports universal family acceptance or a blanket
claim that a model is unsuitable. Preserve all failed and legacy receipts.

G6 resume slice C51 now requires valid recorded SHA-256 identities for local
manifest inputs before a runner can execute; negative tests and existing adapter
controls pass. Next return to the first actionable product/model gap in the
existing task files, auditing current source before acting on an old OPEN entry.
Numerical deferral does not mean
model promotion, and harness progress must still enable usable library delivery.
No new queue, downloads, provider repair or sibling-write authority is introduced.

## Copy-ready goal

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
Read AGENTS.md and docs/AGENT_ENTRYPOINT.md, then
read docs/AGENT_SAFETY_RULES.md, docs/architecture.md,
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
and lifecycle; io for assets; audio for reusable
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
and useful TDT derivatives, including separately evaluated quantized or faster
variants. Parakeet CTC is on hold. Fewer file bytes from fp16/int8/ternary
packaging are not fewer parameters, but unchanged parameter count alone is not
a reason to exclude a measured speed/memory/quality candidate. Compare packed
and expanded weight formats, actual engine/kernel requirements and matched
browser/native performance separately. Investigation of integer/ternary
candidates is authorized; default adoption and acceptance are not. Orukeet is
a roughly 627M-parameter fine-tune, not a small
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
Read the latest handover routed by docs/AGENT_ENTRYPOINT.md and reconcile each required gate with live
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
data. Do not weaken tolerances or erase failed evidence. The user authorized scoped
local commits after validation; preserve unrelated work and never use broad
staging. Pushes, publication, uploads, new large downloads, toolchain/global
installation or external settings changes require separate authority.
Delegate only independent, safely isolated work with clear ownership.

Report PASS, FAIL, PARTIAL, UNTESTED and SKIP honestly, with concrete evidence,
measured improvements, unresolved gates and the next actionable step. Define
finite delivery milestones within this ongoing mission; never declare the
entire library finished because one model or one test lane passes. Continue
safely actionable agreed work, and ask when a material scope or authority
decision is needed. The outcome is a better usable speech library and examples,
verified valuable model support, and a better system for the next port.
```

## Additional confirmed operating constraints

- Limit browser ORT WASM threads to floor(reported hardwareConcurrency / 2) + 2.
  Preserve a lower explicit request and the single-thread fallback where shared
  memory is unavailable. Do not apply this cap to native ORT intra/inter-op settings.
  Inspect src/inference/backends/wasm-threads.ts; do not duplicate the formula.
- OMP is an authorized interactive coding harness for scoped subtasks, not an
  authority to alter global configuration or repair its installation. Root
  AGENTS.md is discovered from the project cwd; explicit startup uses the
  tracked entrypoint. Give each delegate clear ownership and validation gates.
  Use isolated writable checkouts for parallel edits; read-only delegation is
  safe in the shared checkout. Never delegate blind edits to protected changes.
- Keep G0 baseline, G1 models, G2 API, G3 portable/compact runtime, G4 speech
  workflows, G5 examples and G6 engineering as the whole-product tracks.
  Keep P01-P10 as the ordered model lane and C-series product slices in their
  existing files. The handover gives resume priorities, not a competing backlog.
- “Best” is a measured ambition, not permission to pursue unlimited research.
  Keep the initial delivery contract in section 10. Ask before changing required
  outcomes or declaring the whole mission complete.
