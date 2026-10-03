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
   Production feature parity remains FAIL on upstream FFT/power arithmetic;
   the browser real-hybrid score gate remains FAIL and is now attributed to
   real-encoder feed error rather than head or normalization. Both are stated
   with their real causes. Do not relax tolerances to close them, and do not
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

Two gates are open and attributed; nothing here is accepted:

- Production feature parity: FAIL, JFK 313/140,928 mismatches
  (`filterbank-feature-jfk-2026-10-03.json`), upstream FFT/power arithmetic.
- Browser real-hybrid scores: FAIL at head call 20 / frame 69, attributed to
  real-encoder feed error; the WASM head and predictor-state drift are
  exonerated by controlled perturbation. Full-tensor encoder accuracy against
  the official encoder is 0 of 141,312 coordinates failing the unchanged gate.

The next model-port candidate audit is the natural first slice once the
user's authority is confirmed for any new weight download.

## Authority constraints unchanged

Scoped validated local commits are authorized. Push, publish, Hugging Face
mutation, new large weight downloads, global installs or settings changes, and
out-of-scope sibling writes require new authority. Preserve the inherited
Nemotron dirty work and all failed evidence.
