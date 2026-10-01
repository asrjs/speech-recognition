# Shared harness, model-owned contracts

This is an engineering boundary, not a new runtime architecture or a universal
decoder. Follow `architecture.md` and `AGENT_SAFETY_RULES.md`.

## What is shared

Share fixture prerequisites, audio/source identity, numerical comparison,
timeouts, backend identity, run receipts, and orchestration/cleanup conventions.
Raw source-file identity is necessary but does not prove identical effective
model input. For prepared-input comparisons record waveform rate/channels,
float32 encoding/hash, sample and valid lengths, padding and preparation policy.
Compare captured official PCM unchanged to isolate model math; validate the real
source-decoding/resampling path separately. Legacy captures without this boundary
are UNVERIFIED for input identity, not silently promoted by matching output text.
Keep this tooling outside the published runtime. The stage comparator lives in
`tools/model-debugging/scripts/`; Vitest prerequisites live in
`tests/support/fixture-gates.ts` and have model-neutral unit tests plus
cross-model wiring regressions in `tests/fixture-gates-integration.test.ts`.
Partial session acquisition has a separate test-only ownership helper in
`tests/support/fixture-sessions.ts`: track each acquired session, roll back on
setup failure, attempt and await every release, and retain setup/cleanup errors.
Successful groups transfer ownership and provide idempotent awaited cleanup.
It knows no model, graph, provider or decoder contract and is not runtime code.

Model-neutral Python artifact tooling now lives in `tools/onnx_artifacts.py`
(all tensor fields, colocated ranges), `tools/repack_onnx_external_data.py`
(bounded-copy, byte-identity packaging) and `tools/tensor_parity.py`
(full-shape/dtype/finite numerical metrics, exact integer comparisons).
The family-owned NeMo TDT `validate_stage_replay.py` supplies actual frames,
label/state associations and all-channel score normalization. Its native
Python CPU replay PASS does not imply Node/Bun native, browser WebGPU, or
accumulated library transcript acceptance. Neither tools nor test dependencies
are imported into the published runtime.

Integration tests have three distinct states:

- **SKIP:** the opt-in was not enabled / no fixture was requested. Use the test
  runner's skip API; a logged warning followed by `return` is a false PASS.
- **FAIL:** the test was requested but required settings, artifacts or oracle
  data are absent, empty, malformed or incompatible. Validate before creating
  sessions. Per-test prerequisite failures must remain visible in JSON reports.
- **PASS:** the requested test executed its actual assertions. Passing a generic
  gate, simulation or shape smoke is not numerical model or browser parity.

Keep environment changes local to child processes in actual-runner gate checks.
Worker-local unit simulations that replace engine/cache/environment state must
restore the prior state after each case and never load real native bindings.
Generated diagnostics belong under local results/temp storage, not in original
weight/oracle directories. Never silently mutate an oracle to match a candidate.

## What stays model-specific

`src/models/<technical-family>` owns feature-contract glue, tensor/session
wiring, decoder/cache/state behavior and timestamp reconstruction.
`src/presets/<brand>` remains a thin source/configuration layer. Shared tooling
must not import a branded preset to decide how every model should behave.

Each family/reference adapter and its focused tests supply graph names, layouts,
tokenizer/configuration, required stages and their order, output metadata,
tolerances, backend/precision support and exact token/text oracles. Keep NeMo
TDT capture instrumentation in its reference adapter; do not turn TDT durations,
Whisper KV caches or one model's blank/EOS rules into universal decoding rules.
Missing required or one-sided output metadata must fail comparison, not be
discarded to make cross-family comparisons pass. Numerical captures compare the
same model, source/audio and inference contract, not unrelated architectures.

Promote a shared abstraction only after real reuse is demonstrated. Prefer
small adapters and caller-supplied contracts over model-ID branches in a generic
runner, copy-pasted inference loops or a generic utility blob.

## Current implementation and next small steps

The shared fixture gate is used by Whisper merged/split-graph, timestamped
decoder and mel-reference tests, and GigaAM CTC backend tests. Model-specific
assertions and numerical tolerances are unchanged. Machine-specific fixture
roots have configurable/local replacements; see `tests/fixtures/README.md`.

Remaining work, in order as relevant to the active Orukeet port:

1. The Whisper reproducibility harness now narrows tensor metadata and types
   step feeds explicitly; its separate strict test-file compilation passes.
   Reference inference without opt-in is a real SKIP. A requested missing,
   empty or malformed reference and absent mel/audio fail before native session
   creation, covered by five actual-runner wiring regressions. Its existing
   100% feature-input and >=80% WAV-input token criteria are unchanged; this is
   harness correctness, not fresh Whisper inference acceptance. Partial setup
   now uses the ownership helper: nine neutral resource tests and two actual
   Whisper setup simulations cover rollback and retry after a cached rejection.
   Focused tests: 31 passed / 3 skipped; full suite: 1,203 passed / 27 skipped /
   0 failed in `tools/data/results/harness/fixture-session-cleanup-full-suite-2026-10-01.json`.
   Strict changed-test compilation passes. Complete artifact-enabled validation
   separately; simulated ownership checks do not establish native inference.
2. The NeMo inspector's JS/no-processor-ONNX prerequisite is now fixed and covered
   by three artifact-contract regressions (`nemo-inspector-artifacts.test.ts`).
   Direct-family/config/artifact selection and explicit experimental native-CPU
   execution now exist. Node and Bun same-PCM Orukeet output-only receipts are
   recorded; broad native family/provider and browser acceptance remain open.
   The inspector validates captured PCM with `--input-mode oracle-pcm`; do not
   confuse this frozen-input control with source-resampling acceptance. Extend
   the existing family adapter rather than adding another inference pipeline.
3. Make the sibling Parakeet browser runner accept candidate artifacts and an
   audio/token/text oracle rather than only v2/v3 names and a fixed transcript.
   Keep preprocessing and inference in the library; preserve each sibling repo's
   Git state and validate its actual WebGPU/hybrid backend and cleanup.
4. Introduce a declarative capture/output contract when a second actual adapter
   needs it. Reuse numerical metrics, audio identity and receipts, while each
   family declares its required stages and output fields. Preserve fail-closed
   behavior and all existing parity thresholds.

Track completion and evidence in `PARAKEET_VARIANT_TASKS.md` (P09). These are
incremental tasks, not claims that all historical harness hardcodes are fixed.
