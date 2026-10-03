# Goal — model-port scale and harness leverage

Updated: 2026-10-03. Derived from the whole-product goal in
[GOAL_PROMPT_RECOMMENDED.md](GOAL_PROMPT_RECOMMENDED.md) and this session's
measured evidence. This is a focus overlay for **adapting more models and
compounding harness value**, not a replacement for the whole-product goal and
not a claim of current acceptance. Task records remain
[PRODUCT_TASKS.md](PRODUCT_TASKS.md) and
[PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md).

## Why this focus, from evidence

Orukeet consumed many sessions and its remaining gates are now precisely
characterized rather than vague. The bottleneck is no longer "can we port a
model" but "how much harness capability makes the _next_ port cheap". Four
measured facts set this direction:

- The library, harness and evidence discipline work: real native Node/Bun
  oracle-matching output, a real-provider disposal-drain control, and a
  1,462-test suite. Lifecycle ownership is understood and instrumented.
- Attribution is now cheap and decisive. Feeding a recorded encoder delta into
  official reference inputs reproduced a browser gate failure exactly, and
  exonerated the WASM head and predictor-state drift in one experiment.
- The hardest remaining Orukeet failures are **upstream numerical arithmetic**
  (production feature parity: JFK 313/140,928 mismatches). Decoding, state and
  output contracts are already proven.
- Each port has re-derived the same harness primitives: fixture identity,
  oracle loading, stage comparison, lifecycle controls, receipts. That work
  should be reusable, not rewritten per model.

## Goal text

```text
GOAL: Make the next ASR or alignment port materially cheaper than Orukeet was,
and turn the harness work that port forced into reusable, model-neutral
capability — while finishing the Orukeet numerical lane honestly.

Treat the Orukeet port as the reference cost baseline, not as work to repeat.
Its durable lessons are already recorded: official-engine capture bound to
effective float32 PCM, own fp32 ONNX exports, a parity ladder that finds the
EARLIEST divergence, fail-closed controls, and receipts bound to the code and
artifacts that actually executed. Reuse that pattern; do not re-derive it.

1. Close the two known Orukeet numerical gates or reclassify them honestly.
   Production feature parity remains FAIL on upstream FFT/power arithmetic.
   Browser real-hybrid scores now PASS on the documented JFK configuration
   after fixing an ORT-build confound; this is not full ASR acceptance.
   Preserve both gates and their actual scope. Do not relax tolerances, and do not
   keep re-running rejected FFT variants. Finish them only with a genuinely new
   discriminating hypothesis, or record a bounded, evidence-backed deferral.

2. Convert per-port harness knowledge into shared capability. The reusable core
   is: audio/effective-input identity binding, oracle loading and binding,
   stage comparison with per-coordinate tolerances, family-declared required
   stages and output fields, run receipts, and lifecycle/disposal controls.
   Promote a shared abstraction only after real reuse is demonstrated. Keep
   family math, decoding, cache rules and tests in `src/models/<family>`.
   A second adapter needing the same contract is the trigger to generalize —
   not the first.

3. Make attribution cheap for the next port. The technique that worked here is
   reusable: take a captured failing input, replay the component under test
   against official reference inputs, then perturb one input at a time to
   localize the divergent stage. Generalize that into a supported harness
   pattern with fail-closed semantics, so the next port inherits a working
   bisection instead of bespoke investigation.

4. Audit genuine smaller Parakeet students and distilled candidates next, then
   ranked non-NVIDIA ASR/alignment/diarization candidates. Fewer file bytes from
   fp16/int8 packaging is not fewer parameters; Orukeet is ~627M and is not a
   student. For each candidate, verify upstream identity, license,
   architecture, tokenizer, Turkish/multilingual value, memory, latency and
   conversion route before committing effort. Defer with a bounded, evidenced
   explanation, not a claim that a model is bad.

5. Keep the browser lane our differentiator, and keep gates honest. WebGPU is
   primary; WASM is an intentional per-component fallback. Record actual
   host, engine, provider and precision per component. A component is not
   accepted because it passes its own gate while a downstream gate fails for
   upstream reasons; state both facts.

6. Reuse the harness at the boundary it fails. Before writing a new runner for
   a new model or host, check whether an existing adapter, oracle or receipt
   covers it, and extend that instead. Each substantial investigation should
   leave a tested recipe, tool, regression or failure record.

## Anti-goals

- More model names, scripts or abstractions presented as progress.
- Repeating a completed diagnostic because a later document says it is missing.
- Relaxing a tolerance, injecting a fixture value or hiding a model-specific
  fix in shared runtime to make a gate pass.
- Treating a simulated lifecycle test as real provider acceptance.
- Declaring the mission complete because one model or one lane passes.

## Acceptance for this focus

A port is cheaper when a second candidate reaches real output with the existing
harness and no new per-model inference loop, and when a failure is localized to
a stage by a shared control rather than by bespoke investigation. Report
measured cost against the Orukeet baseline: harness changes made, scripts and
manual steps avoided, and gates closed or honestly deferred.
```

## Current measured state

One numerical gate remains open; full ASR acceptance is not established:

- Production feature parity: FAIL, JFK 313/140,928 mismatches at the
  unchanged `atol=1e-5, rtol=1e-4` (`filterbank-feature-jfk-2026-10-03.json`,
  max `3.423691e-4`). Upstream FFT/power arithmetic.
- Browser real-hybrid scores: **PASS on this fixture** with the webgpu-only ORT
  build — `LIVE_HEAD_NUMERICAL_PASS`, 46 calls, `allclose: true`, zero failing
  calls, including head call 20 which failed at every previous real hybrid run.
  Reached by changing which ORT-Web build is loaded, with no tolerance or
  production-math change. This is **not** full ASR acceptance:
  `full_asr` remains `NOT_PROVEN_FEATURE_GATE_UNACCEPTED` because the
  production-feature gate is still FAIL, and it is one fixture, one adapter, one
  configuration.

The root cause of the long-standing `1.1316e-6` gap was the ORT-Web build, and
the library is bundle-agnostic: it imports the bare specifier
`onnxruntime-web`, so the host's import map decides. The old harness rule keyed
on the decoder backend alone, so a WebGPU-encoder/WASM-decoder hybrid received
the all-bundle. The fixed rule selects the webgpu build when either component
uses WebGPU, unless explicitly overridden. That build is both smaller and
measurably closer to the official CPU engine (`3.427267e-7` versus `1.4566e-6`),
and the passing feed-error bracket was near `3.64e-7`.

The lesson recorded for the next port: compare the loaded ORT artifact identity,
not the version string, before attributing any cross-run numerical difference.

Per-frame sensitivity of the head's score gate to encoder feed error is
measured at only one call and one error direction. That bracket is not a
per-frame budget and does not establish any necessary accuracy bound.

Encoder-side decomposition added a native CPU control on the recorded
production-JS features plus a same-session `native(official features)` control.
Separated at the unchanged gate: execution residual `1.326203e-6`, feature
effect `9.797513e-7`, WebGPU-versus-native `5.885959e-7`, official-feature
WebGPU `1.005828e-6` — all passing, with the encoder attenuating the
`3.423691e-4` feature delta by about 320x. Those controls were later shown to
sit behind an ORT-build confound: every cross-run comparison in that phase
disagreed by a constant `1.1316e-6` because the two runners loaded different
`onnxruntime-web` builds. The decomposition values themselves remain valid as
measurements of their own runs.

Goal item 3 is now implemented, not just described:
`tools/stage_attribution.py` is model-neutral, fail-closed tooling that
implicates an input only when its single-input replay both fails the gate and
reproduces the candidate output, returning `ATTRIBUTION_UNRESOLVED` otherwise.
Ten tests cover it, including an interaction counterexample that a
gate-failure-only rule would misattribute. Applied to the real Orukeet case it
localizes the failure to `encoder_outputs` after three binding gates pass. The
next port inherits this instead of re-deriving it.

Item 4 is audited: no smaller Parakeet student was **identified in this
survey** (a top-40 downloads-sorted query plus six included IDs, not an
exhaustive catalogue). Every official NVIDIA Parakeet checked is ~600M or
larger, and the only sub-300M record, `moondream/parakeet-redux` (148.9M), is a
1.58-bit ternary quantization of TDT v3 rather than a student, excluded by the
standing integer-quantization rule. Turkish is the discriminator among the
candidates checked: no Parakeet and none of the previously ranked non-NVIDIA
models cover it, while `Qwen/Qwen3-ASR-0.6B-hf` does (782,426,112 parameters,
30 languages, Apache-2.0). Discovery evidence only for that `-hf` candidate.
The earlier claim that Qwen needs a new family was incorrect: this repository
already implements `qwen-asr` for the official `Qwen/Qwen3-ASR-0.6B` export
chain. Its [porting playbook](../tools/model-debugging/playbooks/qwen3-asr-porting.md)
and [reference guide](../tools/model-debugging/reference/qwen3-asr-0.6b/README.md)
describe existing local ONNX artifacts and an offline library runner.
This is not proof that the `-hf` candidate has the same export contract.
Reconcile those local assets before requesting downloads. The original
PyTorch snapshot recorded in older receipts is currently absent at its
documented path; a fresh official-engine capture remains a separate prerequisite.

The production-feature gate is not closed. The browser real-hybrid score
comparison, which had failed at every previous run, now passes on this fixture —
and its long-standing `1.1316e-6` discrepancy is **fully explained**: the two
runners loaded different `onnxruntime-web` builds. `browser_head_replay.py`'s
import map selects `ort.webgpu.min.mjs` when the server backend is `webgpu` and
`ort.all.bundle.min.mjs` otherwise, and `initOrt()` imports that bare specifier.
Same declared version `1.29.0`, different build and a different WASM sidecar.
With both paths forced onto the same build, the hybrid's encoder output is
**bitwise identical** (`0.000000e+00`) to a direct session path, so there is no
library defect. Direct observation had already established that the hybrid feeds
features byte-identical to the replay's and that its head feed is bitwise equal
to its own encoder output, so the handoff is exact.

The lesson for the next port is recorded: compare the loaded ORT artifact
identity, not the version string, before attributing any cross-run numerical
difference.

A product-relevant lever follows from this. The webgpu-only build sits closer to
the official CPU engine on this fixture (`3.427267e-7`) than the all-bundle does
(`1.4566e-6`), and the measured feed-error bracket put the passing threshold near
`3.64e-7`. Which build the browser path loads therefore matters for the gate
itself, and is under test.

## Measured acceptance against the Orukeet baseline

The focus criterion is measurable: a port is cheaper when a second candidate
reaches real output with the existing harness and no new per-model inference
loop, and a failure is localized by a shared control rather than bespoke
investigation. Measured so far:

- **Harness changes made:** 1 model-neutral module (`tools/stage_attribution.py`)
  and 1 test module, 592 lines total, replacing 154 lines of duplicated probe
  code that was removed. Net reduction in runner code.
- **Scripts avoided:** the same investigation that produced five bespoke probes
  now needs one family adapter (`head_attribution_control.py`) over the shared
  control. A second port using this contract writes one adapter, not a sweep or
  an attribution implementation.
- **Attribution cost:** the head call 20 failure was localized to
  `encoder_outputs` in one shared-control invocation with three binding gates,
  replacing bespoke per-input replay scripts.
- **Gates closed or honestly deferred:** browser real-hybrid scores pass on
  the documented JFK configuration; production feature parity remains FAIL.

The cheaper-second-port acceptance signal is **not yet demonstrated for this
focus**. Existing Qwen, GigaAM, SenseVoice and X-ASR implementations and their
reference tools predate this investigation. Reuse them rather than claiming
that no second model has been ported. Historical output receipts do not establish
current-build parity or cross-family reuse of the new attribution control.

The ORT-build discrepancy and session-options controls above are already
resolved; do not repeat them as the next experiment. For continued work, start
with the [engineering tool map](../tools/README.md#task-to-tool-map) and
project-local [Model Debugging Skill](../tools/model-debugging/SKILL.md).
Check existing original source, reference environments, local graphs and
family runners before selecting the next numerical control or declaring BLOCKED.

## Authority constraints unchanged

Scoped validated local commits are authorized. Push, publish, Hugging Face
mutation, new large weight downloads, global installs or settings changes, and
out-of-scope sibling writes require new authority. Preserve the inherited
Nemotron dirty work and all failed evidence.
