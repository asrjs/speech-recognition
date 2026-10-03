# Agent entrypoint — all coding harnesses

Updated: 2026-10-03. This is the startup router, not another task queue.
Use it with Codex, OMP or another repository-aware harness.

## Read order and authority

1. Read the project-root `AGENTS.md` when present, and
   [AGENT_SAFETY_RULES.md](AGENT_SAFETY_RULES.md) completely before editing.
2. Read [architecture.md](architecture.md),
   [PROJECT_CHARTER.md](PROJECT_CHARTER.md) and
   [GOAL_PROMPT_RECOMMENDED.md](GOAL_PROMPT_RECOMMENDED.md).
3. Read the current [OMP handover](OMP_HANDOVER_2026-10-03.md).
   Reconcile its snapshot against live Git, source, build and receipts.
4. Read [PRODUCT_TASKS.md](PRODUCT_TASKS.md),
   [PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md),
   [MODEL_HARNESS_CONTRACT.md](MODEL_HARNESS_CONTRACT.md) and
   [WORKSPACE_CONTEXT.md](WORKSPACE_CONTEXT.md); then inspect the actual APIs,
   relevant family tests and tool CLI before choosing the first bounded slice.

Latest direct user instructions define the scope. Repository safety and current
architecture govern implementation. The goal defines durable outcomes; existing
task files hold the work records; the dated handover is evidence, not permanent
certification. Resolve contradictory records against actual source/artifacts and
ask before materially changing the model order or required acceptance gates.

This is one speech-first, ESM-first, headless TypeScript package, not a
Transformers.js clone. Public contracts are coherent and root-exported with
intentional narrow subpaths. Actual folders include `src/audio` and `src/io`,
not `src/processors`. Family math, decoder/cache rules and model-specific tests
stay in `src/models`; brand configurations stay in `src/presets`. Framework
components/state stay outside core. Never move a one-model workaround into shared
runtime merely to get a transcript. Preserve canonical output semantics.

## Historical records are not startup instructions

`GOAL_PROMPT.md`, `HANDOVER.md`, `HANDOFF.md`, `SESSION_HANDOVER.md`,
`AGENT_TASKS.md`, `AGENT_STATUS_REVIEW.md`, `WSL2_PROJECT_HANDOFF.md` and the
archived recommended goal retain useful history. Their old branches, agent
names, resume prompts, queue entries and PASS claims do not direct current work.
Follow them only when the user explicitly resumes that historical scope, after
checking source/build/artifact freshness. Do not check out an old branch because
a historical handoff says to do so. The current model order is Orukeet, then
genuine smaller Parakeet students/distills, then ranked non-NVIDIA candidates;
Nemotron remains paused.

## Portable startup

Start OMP from the project directory. Installed OMP 18.4.12 discovers standalone
`AGENTS.md` up from cwd to the repository boundary. The local root `AGENTS.md`
is ignored by this checkout's Git rules, so it is not guaranteed in a new clone.
This tracked entrypoint is the explicit portable bootstrap. Do not invent OMP
configuration files or change global provider settings.

Give the harness this prompt:

```text
Read docs/AGENT_ENTRYPOINT.md and its required documents completely. Use
docs/GOAL_PROMPT_RECOMMENDED.md as the whole-product goal and reconcile
docs/OMP_HANDOVER_2026-10-03.md with live Git/source/build/artifacts. Preserve
protected dirty work and failed evidence. Resume the first safely actionable
bounded slice in the existing task files, validate it, then commit only owned
changes. Do not resume Nemotron, relax parity gates, repeat completed controls,
push/publish/upload, install globally or download large weights without authority.
```

## Current focus overlay

[GOAL_MODEL_PORT_HARNESS.md](GOAL_MODEL_PORT_HARNESS.md) is derived from that
goal and is the current emphasis: making the next model port cheaper and
compounding harness value. It narrows focus; it does not replace the
whole-product goal, waive any gate or change the confirmed model order.
